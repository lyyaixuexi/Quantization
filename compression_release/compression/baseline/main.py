import argparse

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from compression.baseline.option import Option
from compression.baseline.trainer import QTrainer
from compression.checkpoint import CheckPoint
from compression.dataloader import *
from compression.model_builder import get_model
from compression.utils.logger import get_logger
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

        # self.logger.info('Test')
        # assert False

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

        elif "imagenet" in self.settings.dataset:
            transfroms_name = "default"
            if "mobilenetv3" in self.settings.net_type:
                transfroms_name = "mobilenetv3"
            elif "mobilenet" in self.settings.net_type:
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

        quantize_first_last = self.settings.quantize_first_last
        self.model, _ = get_model(
            self.settings.dataset,
            self.settings.net_type,
            self.settings.depth,
            self.settings.n_classes,
            self.settings
        )
        self.logger.info(self.model)

    def _set_checkpoint(self):
        """
        Load pre-trained model or resume checkpoint
        """

        assert self.model is not None, "please create model first"

        self.checkpoint = CheckPoint(self.settings.save_path, self.logger)
        self._load_pretrained()
        self._load_resume()

    def _load_pretrained(self):
        """
        Load pretrained model
        the pretrained model contains state_dicts of model
        """

        if self.settings.pretrained is not None:
            check_point_params = torch.load(self.settings.pretrained, map_location='cpu')
            model_state = check_point_params
            if self.settings.net_type in ["mobilenetv1"]:
                model_state = check_point_params['model']
            self.model = self.checkpoint.load_state(self.model, model_state)
            self.logger.info(
                "|===>load restrain file: {}".format(self.settings.pretrained)
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

    def _set_trainier(self):
        """
        Initialize trainer for AuxNet
        """

        self.trainer = QTrainer(
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

        for epoch in range(start_epoch, self.settings.n_epochs):
            if self.settings.distributed:
                self.train_sampler.set_epoch(epoch)

            train_error, train_loss, train5_error = self.trainer.train(epoch)
            val_error, val_loss, val5_error = self.trainer.val(epoch)

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
                # if self.settings.dataset in ["imagenet"]:
                #     self.checkpoint.save_checkpoint(
                #         self.model,
                #         self.trainer.optimizer,
                #         self.trainer.scheduler,
                #         epoch,
                #         epoch,
                #     )
                # else:
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
    args = parser.parse_args()

    option = Option(args.conf_path)
    option.manualSeed = args.id + 1
    option.experiment_id = option.experiment_id + "{:0>2d}".format(args.id + 1)

    experiment = Experiment(option)
    experiment.run()


if __name__ == "__main__":
    main()
