import argparse

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from compression.checkpoint import CheckPoint
from compression.dataloader import *
from compression.model_builder import get_model
from compression.models.quantization.qasy_mobilenetv3_cifar import QAsyMobileBlockCifar
from compression.models.quantization.qmobilenetv2 import QMobileBottleneck
from compression.models.quantization.qpreresnet import QPreBasicBlock
from compression.models.quantization.qresnet import QBasicBlock, QBottleneck
from compression.models.quantization.super_prune_gate_preresnet import (
    SuperPrunedGatePreBasicBlock,
)
from compression.single_path_compress_binary_gate.option import Option
from compression.trainer import Trainer
from compression.utils.logger import get_logger
from compression.utils.model_analyse import ModelAnalyse
from compression.utils.pruning import ResModelPrune
from compression.utils.qmodel_analyse import QModelAnalyse
from compression.utils.utils import *
from compression.utils.write_log import write_log, write_settings
from torch.utils.tensorboard import SummaryWriter


class Experiment(object):
    """
    Run experiments with pre-defined pipeline
    """

    def __init__(self, options=None, conf_path=None):
        self.settings = options or Option(conf_path)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.settings.gpu
        self.settings.gpu = 0
        init_distributed_mode(self.settings)
        self.checkpoint = None
        self.train_loader = None
        self.val_loader = None
        self.model = None

        self.trainer = None
        self.optimizer_state = None
        self.lr_scheduler_state = None

        self.settings.set_save_path()
        if is_main_process():
            write_settings(self.settings)
        if self.settings.distributed:
            torch.distributed.barrier()

        self.logger = get_logger(self.settings.save_path, "baseline")
        setup_logger_for_distributed(self.settings.rank == 0, self.logger)
        self.tensorboard_logger = SummaryWriter(self.settings.save_path)
        setup_tensorboard_logger_for_distributed(
            self.settings.rank == 0, self.tensorboard_logger
        )

        self.settings.copy_code(
            self.logger,
            src=os.path.abspath("../"),
            dst=os.path.join(self.settings.save_path, "code"),
        )

        self.logger.info(
            "|===>Result will be saved at {}".format(self.settings.save_path)
        )

        self.start_epoch = 0
        self.device = None

        self.prepare()

    def prepare(self):
        """
        Prepare experiments
        """

        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._set_checkpoint()
        self._pruning()
        self._load_resume()
        self._set_trainier()

    def _set_gpu(self):
        """
        Initialize the seed of random number generator
        """

        # set torch seed
        # init random seed
        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed(self.settings.seed)
        torch.cuda.set_device(self.settings.gpu)
        self.device = torch.device("cuda")
        cudnn.benchmark = True

    def _set_dataloader(self):
        """
        Create train loader and validation loader for auxnet
        """
        if "cifar" in self.settings.dataset:
            self.train_loader, self.val_loader = get_cifar_dataloader(
                self.settings.dataset,
                self.settings.batch_size,
                self.settings.n_threads,
                self.settings.data_path,
                self.settings.distributed,
                self.logger,
            )
            self.logger.info("Test")

        elif "imagenet" in self.settings.dataset:
            transfroms_name = "default"
            if "mobilenet" in self.settings.net_type:
                transfroms_name = "mobilenet"

            (
                self.train_loader,
                self.val_loader,
                self.train_sampler,
                self.val_sampler,
            ) = get_imagenet_dataloader(
                self.settings.dataset,
                self.settings.batch_size,
                self.settings.n_threads,
                self.settings.data_path,
                transfroms_name,
                self.settings.distributed,
                self.logger,
            )

    def _set_model(self):
        """
        Available model
        cifar:
            preresnet
            vgg
        imagenet:
            resnet
        """

        self.model, self.test_input = get_model(
            self.settings.dataset,
            self.settings.net_type,
            self.settings.depth,
            self.settings.n_classes,
            self.settings,
        )
        self.logger.info(self.model)

    def _set_checkpoint(self):
        """
        Load pre-trained model or resume checkpoint
        """

        assert self.model is not None, "please create model first"

        self.checkpoint = CheckPoint(self.settings.save_path, self.logger)
        self._load_pretrained()
        self._load_fp_weight()

    def _load_pretrained(self):
        """
        Load pretrained model
        the pretrained model contains state_dicts of model
        """

        if self.settings.pretrained is not None:
            check_point_params = torch.load(
                self.settings.pretrained, map_location="cpu"
            )
            model_state = check_point_params
            model_state = check_point_params["model"]

            if self.settings.net_type in ["qpreresnet", "qresnet"]:
                bit_key_names = ["bits_weights", "bits_activations"]
                layer_key_names = ["conv1", "conv2", "downsample"]
                for name, module in self.model.named_modules():
                    if isinstance(module, (QPreBasicBlock, QBasicBlock)):
                        for layer_key_name in layer_key_names:
                            for bit_key_name in bit_key_names:
                                if (
                                    layer_key_name == "downsample"
                                    and module.downsample is None
                                ):
                                    continue
                                elif (
                                    layer_key_name == "downsample"
                                    and module.downsample is not None
                                ):
                                    if self.settings.net_type == "qresnet":
                                        key_name = "{}.{}.0.{}".format(
                                            name, layer_key_name, bit_key_name
                                        )
                                        value = model_state[key_name]
                                        setattr(
                                            getattr(
                                                module, "{}".format(layer_key_name)
                                            )[0],
                                            "{}".format(bit_key_name),
                                            value.item(),
                                        )
                                    else:
                                        key_name = "{}.{}.{}".format(
                                            name, layer_key_name, bit_key_name
                                        )
                                        value = model_state[key_name]
                                        setattr(
                                            getattr(
                                                module, "{}".format(layer_key_name)
                                            ),
                                            "{}".format(bit_key_name),
                                            value.item(),
                                        )
                                else:
                                    key_name = "{}.{}.{}".format(
                                        name, layer_key_name, bit_key_name
                                    )
                                    value = model_state[key_name]
                                    setattr(
                                        getattr(module, "{}".format(layer_key_name)),
                                        "{}".format(bit_key_name),
                                        value.item(),
                                    )

                        module.channel_thresholds = nn.Parameter(torch.zeros(1))
                        module.group_size = self.settings.group_size
                        module.register_buffer(
                            "assigned_indicator",
                            torch.zeros(module.conv1.out_channels // module.group_size),
                        )
                    elif isinstance(module, QBottleneck):
                        self.logger.info("Loading resnet50 checkpoint!")
                        layer_key_names = ["conv1", "conv2", "conv3", "downsample"]
                        for layer_key_name in layer_key_names:
                            for bit_key_name in bit_key_names:
                                if (
                                    layer_key_name == "downsample"
                                    and module.downsample is None
                                ):
                                    continue
                                elif (
                                    layer_key_name == "downsample"
                                    and module.downsample is not None
                                ):
                                    key_name = "{}.{}.0.{}".format(
                                        name, layer_key_name, bit_key_name
                                    )
                                    value = model_state[key_name]
                                    setattr(
                                        getattr(module, "{}".format(layer_key_name))[0],
                                        "{}".format(bit_key_name),
                                        value.item(),
                                    )
                                else:
                                    key_name = "{}.{}.{}".format(
                                        name, layer_key_name, bit_key_name
                                    )
                                    value = model_state[key_name]
                                    setattr(
                                        getattr(module, "{}".format(layer_key_name)),
                                        "{}".format(bit_key_name),
                                        value.item(),
                                    )

                        module.group_size = self.settings.group_size
                        module.channel_thresholds_1 = nn.Parameter(torch.zeros(1))
                        module.register_buffer(
                            "assigned_indicator_1",
                            torch.zeros(module.conv1.out_channels // module.group_size),
                        )

                        module.channel_thresholds_2 = nn.Parameter(torch.zeros(1))
                        module.register_buffer(
                            "assigned_indicator_2",
                            torch.zeros(module.conv1.out_channels // module.group_size),
                        )

            elif self.settings.net_type in ["mobilenetv2", "qmobilenetv2"]:
                bit_key_names = ["bits_weights", "bits_activations"]
                layer_key_names = ["conv1", "conv2", "conv3"]
                for name, module in self.model.named_modules():
                    if isinstance(module, (QMobileBottleneck)):
                        module.channel_thresholds = nn.Parameter(torch.zeros(1))
                        module.group_size = self.settings.group_size
                        module.register_buffer(
                            "assigned_indicator",
                            torch.zeros(module.conv2.out_channels // module.group_size),
                        )

                        for layer_key_name in layer_key_names:
                            if module.expand == 1 and layer_key_name == "conv1":
                                continue
                            else:
                                for bit_key_name in bit_key_names:
                                    key_name = "{}.{}.{}".format(
                                        name, layer_key_name, bit_key_name
                                    )
                                    value = model_state[key_name]
                                    setattr(
                                        getattr(module, "{}".format(layer_key_name)),
                                        "{}".format(bit_key_name),
                                        value.item(),
                                    )

                layer_key_name = "conv2"
                for bit_key_name in bit_key_names:
                    key_name = "{}.{}".format(layer_key_name, bit_key_name)
                    value = model_state[key_name]
                    setattr(
                        getattr(self.model, "{}".format(layer_key_name)),
                        "{}".format(bit_key_name),
                        value.item(),
                    )

            elif self.settings.net_type in ["qasymobilenetv3_cifar"]:
                bit_key_names = ["bits_weights", "bits_activations"]
                layer_key_names = ["conv1", "conv2", "conv3"]
                for name, module in self.model.named_modules():
                    if isinstance(module, (QAsyMobileBlockCifar)):
                        module.channel_thresholds = nn.Parameter(torch.zeros(1))
                        module.group_size = self.settings.group_size
                        module.register_buffer(
                            "assigned_indicator",
                            torch.zeros(module.conv2.out_channels // module.group_size),
                        )
                        for layer_key_name in layer_key_names:
                            if not module.expand and layer_key_name == "conv1":
                                continue
                            else:
                                for bit_key_name in bit_key_names:
                                    key_name = "{}.{}.{}".format(
                                        name, layer_key_name, bit_key_name
                                    )
                                    value = model_state[key_name]
                                    setattr(
                                        getattr(module, "{}".format(layer_key_name)),
                                        "{}".format(bit_key_name),
                                        value.item(),
                                    )

                        if hasattr(module, "squeeze_block"):
                            se_layers = [0, 2]
                            for se_layer in se_layers:
                                for bit_key_name in bit_key_names:
                                    key_name = "{}.squeeze_block.dense.{}.{}".format(
                                        name, se_layer, bit_key_name
                                    )
                                    value = model_state[key_name]
                                    setattr(
                                        module.squeeze_block.dense[se_layer],
                                        "{}".format(bit_key_name),
                                        value.item(),
                                    )

                layer_key_name = "conv2"
                for bit_key_name in bit_key_names:
                    key_name = "{}.{}".format(layer_key_name, bit_key_name)
                    value = model_state[key_name]
                    setattr(
                        getattr(self.model, "{}".format(layer_key_name)),
                        "{}".format(bit_key_name),
                        value.item(),
                    )

                layer_key_name = "conv3"
                for bit_key_name in bit_key_names:
                    key_name = "{}.{}".format(layer_key_name, bit_key_name)
                    value = model_state[key_name]
                    setattr(
                        getattr(self.model, "{}".format(layer_key_name)),
                        "{}".format(bit_key_name),
                        value.item(),
                    )

            self.model = self.checkpoint.load_state(self.model, model_state)
            self.logger.info(
                "|===>load restrain file: {}".format(self.settings.pretrained)
            )

    def _load_fp_weight(self):
        if self.settings.mixed_precision_pretrained is not None:
            check_point_params = torch.load(
                self.settings.mixed_precision_pretrained, map_location="cpu"
            )
            model_state = check_point_params
            # model_state = check_point_params['model']
            self.model = self.checkpoint.load_state(self.model, model_state)
            self.logger.info(
                "|===>load restrain file: {}".format(
                    self.settings.mixed_precision_pretrained
                )
            )

    def _load_resume(self):
        """
        Load resume checkpoint
        the checkpoint contains state_dicts of model and optimizer, as well as training epoch
        """

        if self.settings.resume is not None:
            (
                model_state,
                optimizer_state,
                epoch,
                lr_scheduler_state,
            ) = self.checkpoint.load_checkpoint(self.settings.resume)
            self.model = self.checkpoint.load_state(self.model, model_state)
            self.start_epoch = epoch
            self.optimizer_state = optimizer_state
            self.lr_scheduler_state = lr_scheduler_state
            self.logger.info("|===>load resume file: {}".format(self.settings.resume))

    def _pruning(self):
        if self.settings.net_type in [
            "preresnet",
            "qpreresnet",
            "qresnet",
            "resnet",
            "qmobilenetv2",
            "qasymobilenetv3_cifar",
        ]:
            model_prune = ResModelPrune(
                model=self.model,
                net_type=self.settings.net_type,
                depth=self.settings.depth,
            )
        else:
            assert False, "unsupport net_type: {}".format(self.settings.net_type)

        model_prune.run()
        self.model = model_prune.model
        self.logger.info("After pruning:")
        self.logger.info(self.model)

        if self.settings.net_type in [
            "qpreresnet",
            "qresnet",
            "qmobilenetv2",
            "qasymobilenetv3_cifar",
        ]:
            model_analyse = QModelAnalyse(self.model, self.logger)
        else:
            model_analyse = ModelAnalyse(self.model, self.logger)
        model_analyse.bops_compute(self.test_input)
        # assert False

    def _set_trainier(self):
        """
        Initialize trainer for AuxNet
        """

        self.trainer = Trainer(
            model=self.model,
            device=self.device,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            settings=self.settings,
            logger=self.logger,
            tensorboard_logger=self.tensorboard_logger,
            optimizer_state=self.optimizer_state,
            lr_scheduler_state=self.lr_scheduler_state,
            run_count=self.start_epoch,
        )

    def get_channel(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (SuperPrunedGatePreBasicBlock)):
                # self.logger.info('Layer: {}'.format(name))
                module.compute_indicator()
                num_group = module.indicator.sum().item()
                channel_num = num_group * module.group_size
                self.logger.info("Layer: {}, channel: {}".format(name, channel_num))
                filter_weight = module.conv1.weight
                normalized_filter_weight_norm = module.compute_norm(filter_weight)
                self.logger.info(normalized_filter_weight_norm)

    def run(self):
        """
        Learn the parameters of the additional classifier and
        fine tune model with the additional losses and the final loss
        """

        best_top1 = 100
        best_top5 = 100

        start_epoch = 0
        # if load resume checkpoint
        if self.start_epoch != 0:
            start_epoch = self.start_epoch + 1
            self.epoch = 0

        val_error, val_loss, val5_error = self.trainer.val_without_tb(0)
        # assert False

        if self.settings.from_scratch:
            self.logger.info("Reset parameters!")
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    m.reset_parameters()

        for epoch in range(start_epoch, self.settings.n_epochs):
            if self.settings.distributed:
                self.train_sampler.set_epoch(epoch)

            # train_error, train_loss, train5_error = self.trainer.train(epoch)
            train_error, train_loss, train5_error = self.trainer.train(epoch)
            val_error, val_loss, val5_error = self.trainer.val(epoch)
            # self.trainer.get_channels()

            # write log
            log_str = "{:d}\t".format(epoch)
            log_str += "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(
                train_error, train_loss, val_error, val_loss, train5_error, val5_error
            )
            if self.settings.rank == 0:
                write_log(self.settings.save_path, "log.txt", log_str)

            # save model and checkpoint
            best_flag = False
            if best_top1 >= val_error:
                best_top1 = val_error
                best_top5 = val5_error
                best_flag = True

            self.logger.info(
                "|===>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}\n".format(
                    best_top1, best_top5
                )
            )
            self.logger.info(
                "|==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}\n".format(
                    100 - best_top1, 100 - best_top5
                )
            )

            if self.settings.rank == 0:
                if self.settings.dataset in ["imagenet"]:
                    self.checkpoint.save_checkpoint(
                        self.model,
                        self.trainer.optimizer,
                        self.trainer.scheduler,
                        epoch,
                        epoch,
                    )
                else:
                    self.checkpoint.save_checkpoint(
                        self.model,
                        self.trainer.optimizer,
                        self.trainer.scheduler,
                        epoch,
                    )

                if best_flag:
                    self.checkpoint.save_model(self.model, best_flag=best_flag)


def main():
    parser = argparse.ArgumentParser(description="Baseline")
    parser.add_argument(
        "conf_path",
        type=str,
        metavar="conf_path",
        help="the path of config file for training (default: 64)",
    )
    parser.add_argument("id", type=int, metavar="experiment_id", help="Experiment ID")
    parser.add_argument("gpu_id", type=str, metavar="gpu id", help="gpu id")
    args = parser.parse_args()

    option = Option(args.conf_path)
    option.manualSeed = args.id + 1
    option.experiment_id = option.experiment_id + "{:0>2d}".format(args.id + 1)
    option.gpu = args.gpu_id

    experiment = Experiment(option)
    experiment.run()


if __name__ == "__main__":
    main()
