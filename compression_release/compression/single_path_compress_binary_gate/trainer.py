import torch
import torch.nn as nn
from compression.checkpoint import CheckPoint
from compression.models.quantization.dorefa_clip import QConv2d, QLinear
from compression.models.quantization.super_compress_gate_mobilenetv2 import (
    SuperCompressedMobileBottleneck,
)
from compression.models.quantization.super_compress_gate_mobilenetv3_cifar import (
    SuperCompressAsyMobileBlockCifar,
)
from compression.models.quantization.super_compress_gate_preresnet import (
    SuperCompressedGatePreBasicBlock,
)
from compression.models.quantization.super_compress_gate_resnet import (
    SuperCompressedGateBasicBlock,
    SuperCompressedGateBottleneck,
)
from compression.models.quantization.super_compress_gate_resnet_group import (
    SuperCompressedGateBasicBlockGroup,
    SuperCompressedGateBottleneckGroup,
    SuperCompressedGateResNetGroup,
)
from compression.models.quantization.super_quan_asy_conv import (
    SuperAsyQConv2d,
    SuperAsyQLinear,
)
from compression.models.quantization.super_quan_conv import (
    SuperQConv2d,
    SuperQLinear,
    SuperQWeightConv2d,
)
from compression.trainer import Trainer
from compression.utils.scheduler import GradualWarmupScheduler
from compression.utils.utils import *
from compression.utils.utils import compute_bops
from numpy import isin


class QTrainer(Trainer):
    """
    Trainer for auxnet
    """

    def __init__(
        self,
        model,
        device,
        train_loader,
        val_loader,
        settings,
        logger,
        tensorboard_logger,
        optimizer_state=None,
        lr_scheduler_state=None,
        run_count=0,
    ):
        self.settings = settings

        self.device = device
        self.model = model
        self.model = self.model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.logger = logger
        self.tensorboard_logger = tensorboard_logger
        self.run_count = run_count
        self.iteration_checkpoint = CheckPoint(self.settings.save_path, logger)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = self.settings.lr
        network_params = []
        for name, param in self.model.named_parameters():
            weight_decay = self.settings.weight_decay
            if "channel_thresholds" in name:
                network_params.append(
                    {"params": param, "lr": self.settings.lr, "weight_decay": 0}
                )
            elif (
                "weight_quantization_thresholds" in name
                or "activation_quantization_thresholds" in name
            ):
                network_params.append(
                    {"params": param, "lr": self.settings.lr, "weight_decay": 0}
                )
            else:
                # param.requires_grad = False
                network_params.append(
                    {
                        "params": param,
                        "lr": self.settings.lr,
                        "weight_decay": weight_decay,
                    }
                )
        if "SGD" in self.settings.opt_type:
            self.optimizer = torch.optim.SGD(
                params=network_params,
                lr=self.settings.lr,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weight_decay,
                nesterov=True,
            )
        elif "Adam" in self.settings.opt_type:
            self.optimizer = torch.optim.Adam(
                params=network_params,
                lr=self.settings.lr,
                weight_decay=self.settings.weight_decay,
            )

        self.logger.info(self.optimizer)
        # assert False

        if optimizer_state is not None:
            self.logger.info("Load optimizer state!")
            self.optimizer.load_state_dict(optimizer_state)

        if "cosine" in self.settings.lr_scheduler_type:
            self.logger.info("Cosine Annealing LR!")
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.settings.n_epochs
            )
        elif "multi_step_warmup" in self.settings.lr_scheduler_type:
            self.logger.info("MultiStep Warmup LR!")
            self.after_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.settings.step, gamma=0.1
            )
            self.scheduler = GradualWarmupScheduler(
                self.optimizer, 1, self.settings.warmup_n_epochs, self.after_scheduler
            )
        else:
            self.logger.info("MultiStep LR!")
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.settings.step, gamma=0.1
            )

        if lr_scheduler_state is not None:
            self.logger.info("Load lr state")
            last_epoch = lr_scheduler_state["last_epoch"]
            self.logger.info(self.scheduler.last_epoch)
            while self.scheduler.last_epoch < last_epoch:
                self.scheduler.step()

        model_without_ddp = self.model
        if self.settings.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.settings.gpu], find_unused_parameters=True
            )
            self.model_without_ddp = self.model.module

        self.layer_total_bops = {}
        self.layer_bops = {}
        # self.layer_bitw = {}
        # self.layer_bita = {}
        self.compute_total_bops()

    def forward_hook(self, module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        _, _, w, h = output.shape
        module.w = w
        module.h = h

    def compute_total_bops(self):
        self.hook_list = []

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                self.hook_list.append(module.register_forward_hook(self.forward_hook))

        if "cifar" in self.settings.dataset:
            random_input = torch.randn(1, 3, 32, 32).cuda()
        else:
            random_input = torch.randn(1, 3, 224, 224).cuda()
        self.model(random_input)

        self.total_bops = 0

        if self.settings.net_type in "super_compress_gate_preresnet":
            layer = self.model.conv
            bops = compute_bops(
                layer.kernel_size[0],
                layer.in_channels,
                layer.out_channels,
                layer.h,
                layer.w,
            )
            self.total_bops += bops
            self.layer_total_bops["conv"] = bops

            for i in range(3):
                layer = getattr(self.model, "layer{}".format(i + 1))
                for name, module in layer.named_modules():
                    if isinstance(module, SuperCompressedGatePreBasicBlock):
                        bops = module.get_total_bops()
                        self.layer_total_bops[name] = bops
                        self.total_bops += bops

            layer = self.model.fc
            bops = compute_bops(1, layer.in_features, layer.out_features, 1, 1)
            self.total_bops += bops
            self.layer_total_bops["fc"] = bops

        elif self.settings.net_type in [
            "super_compress_gate_resnet",
            "super_compress_gate_resnet_group",
        ]:
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                layer = self.model.module.conv1
            else:
                layer = self.model.conv1
            bops = compute_bops(
                layer.kernel_size[0],
                layer.in_channels,
                layer.out_channels,
                layer.h,
                layer.w,
            )
            self.total_bops += bops
            self.layer_total_bops["conv"] = bops

            for i in range(4):
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    layer = getattr(self.model.module, "layer{}".format(i + 1))
                else:
                    layer = getattr(self.model, "layer{}".format(i + 1))
                for name, module in layer.named_modules():
                    if isinstance(
                        module,
                        (
                            SuperCompressedGateBasicBlock,
                            SuperCompressedGateBottleneck,
                            SuperCompressedGateBasicBlockGroup,
                            SuperCompressedGateBottleneckGroup,
                        ),
                    ):
                        bops = module.get_total_bops()
                        self.layer_total_bops[name] = bops
                        self.total_bops += bops

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                layer = self.model.module.fc
            else:
                layer = self.model.fc
            bops = compute_bops(1, layer.in_features, layer.out_features, 1, 1)
            self.total_bops += bops
            self.layer_total_bops["fc"] = bops

        elif self.settings.net_type in "super_compress_gate_mobilenetv2":
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                layer = self.model.module.conv1
            else:
                layer = self.model.conv1
            bops = compute_bops(
                layer.kernel_size[0],
                layer.in_channels,
                layer.out_channels,
                layer.h,
                layer.w,
            )
            self.total_bops += bops
            self.layer_total_bops["conv"] = bops
            self.logger.info("Layer: {}, Bops: {}".format("conv1", bops))

            for i in range(7):
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    layer = getattr(self.model.module, "layer{}".format(i + 1))
                else:
                    layer = getattr(self.model, "layer{}".format(i + 1))
                for name, module in layer.named_modules():
                    if isinstance(module, (SuperCompressedMobileBottleneck)):
                        bops = module.get_total_bops()
                        self.layer_total_bops[name] = bops
                        self.total_bops += bops
                        self.logger.info(
                            "Layer: layer{}.{}, Bops: {}".format(i, name, bops)
                        )

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                layer = self.model.module.conv2
            else:
                layer = self.model.conv2
            bops = compute_bops(
                layer.kernel_size[0],
                layer.in_channels,
                layer.out_channels,
                layer.h,
                layer.w,
            )
            self.total_bops += bops
            self.layer_total_bops["conv2"] = bops
            self.logger.info("Layer: {}, Bops: {}".format("conv2", bops))

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                layer = self.model.module.fc
            else:
                layer = self.model.fc
            bops = compute_bops(1, layer.in_features, layer.out_features, 1, 1)
            self.total_bops += bops
            self.layer_total_bops["fc"] = bops
            self.logger.info("Layer: {}, Bops: {}".format("fc", bops))

        elif self.settings.net_type in "super_compress_qsy_mobilenetv3_cifar":
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    bops = compute_bops(
                        module.kernel_size[0],
                        module.in_channels,
                        module.out_channels // module.groups,
                        module.h,
                        module.w,
                    )
                    self.total_bops += bops

                if isinstance(module, nn.Linear):
                    bops = compute_bops(
                        1, module.in_features, module.out_features, 1, 1
                    )
                    self.total_bops += bops

        for hook in self.hook_list:
            hook.remove()

        self.logger.info("Total BOPs: {}M".format(self.total_bops / 1e6))

    def compute_bops_loss(self, iteration=0):
        current_bops = 0

        if self.settings.net_type == "super_compress_gate_preresnet":
            layer = self.model.conv
            bops = compute_bops(
                layer.kernel_size[0],
                layer.in_channels,
                layer.out_channels,
                layer.h,
                layer.w,
                8,
                8,
            )
            # bops = layer.fix_activation_compute_weight_bops()
            current_bops += bops
            self.layer_bops["conv"] = bops

            for i in range(3):
                layer = getattr(self.model, "layer{}".format(i + 1))
                for name, module in layer.named_modules():
                    if isinstance(module, SuperCompressedGatePreBasicBlock):
                        if iteration % 2 == 0:
                            bops = module.compress_fix_activation_compute_weight_bops()
                        else:
                            bops = module.compress_fix_weight_compute_activation_bops()
                        # elif iteration % 3 == 2:
                        # bops = module.compute_bops()
                        self.layer_bops[name] = bops
                        current_bops += bops

            layer = self.model.fc
            bops = compute_bops(1, layer.in_features, layer.out_features, 1, 1, 8, 8)
            # if iteration % 2 == 0:
            #     bops = layer.fix_activation_compute_weight_bops()
            # else:
            #     bops = layer.fix_weight_compute_activation_bops()
            current_bops += bops
            self.layer_bops["fc"] = bops

        elif self.settings.net_type in [
            "super_compress_gate_resnet",
            "super_compress_gate_resnet_group",
        ]:
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                layer = self.model.module.conv1
            else:
                layer = self.model.conv1
            bops = compute_bops(
                layer.kernel_size[0],
                layer.in_channels,
                layer.out_channels,
                layer.h,
                layer.w,
                8,
                8,
            )
            current_bops += bops
            self.layer_bops["conv"] = bops

            for i in range(4):
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    layer = getattr(self.model.module, "layer{}".format(i + 1))
                else:
                    layer = getattr(self.model, "layer{}".format(i + 1))
                for name, module in layer.named_modules():
                    if isinstance(
                        module,
                        (
                            SuperCompressedGateBasicBlock,
                            SuperCompressedGateBottleneck,
                            SuperCompressedGateBasicBlockGroup,
                            SuperCompressedGateBottleneckGroup,
                        ),
                    ):
                        if iteration % 2 == 0:
                            bops = module.compress_fix_activation_compute_weight_bops()
                        else:
                            bops = module.compress_fix_weight_compute_activation_bops()
                        # elif iteration % 3 == 2:
                        # bops = module.compute_bops()
                        self.layer_bops[name] = bops
                        current_bops += bops

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                layer = self.model.module.fc
            else:
                layer = self.model.fc
            bops = compute_bops(1, layer.in_features, layer.out_features, 1, 1, 8, 8)
            current_bops += bops
            self.layer_bops["fc"] = bops

        elif self.settings.net_type in "super_compress_gate_mobilenetv2":
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                layer = self.model.module.conv1
            else:
                layer = self.model.conv1
            bops = compute_bops(
                layer.kernel_size[0],
                layer.in_channels,
                layer.out_channels,
                layer.h,
                layer.w,
                8,
                8,
            )
            current_bops += bops
            self.layer_bops["conv"] = bops

            for i in range(7):
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    layer = getattr(self.model.module, "layer{}".format(i + 1))
                else:
                    layer = getattr(self.model, "layer{}".format(i + 1))
                for name, module in layer.named_modules():
                    if isinstance(module, (SuperCompressedMobileBottleneck)):
                        if iteration % 2 == 0:
                            bops = module.compress_fix_activation_compute_weight_bops()
                        else:
                            bops = module.compress_fix_weight_compute_activation_bops()
                        # elif iteration % 3 == 2:
                        # bops = module.compute_bops()
                        self.layer_bops[name] = bops
                        current_bops += bops

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                layer = self.model.module.conv2
            else:
                layer = self.model.conv2
            if iteration % 2 == 0:
                bops = layer.compress_fix_activation_compute_weight_bops(
                    layer.out_channels, is_out_channel=True
                )
            else:
                bops = layer.compress_fix_weight_compute_activation_bops(
                    layer.out_channels, is_out_channel=True
                )
            current_bops += bops
            self.layer_bops["conv2"] = bops

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                layer = self.model.module.fc
            else:
                layer = self.model.fc
            bops = compute_bops(1, layer.in_features, layer.out_features, 1, 1, 8, 8)
            current_bops += bops
            self.layer_bops["fc"] = bops

        elif self.settings.net_type in "super_compress_qsy_mobilenetv3_cifar":
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                layer = self.model.module.conv1
            else:
                layer = self.model.conv1
            bops = compute_bops(
                layer.kernel_size[0],
                layer.in_channels,
                layer.out_channels,
                layer.h,
                layer.w,
                8,
                8,
            )
            current_bops += bops
            self.layer_bops["conv"] = bops

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                layer = getattr(self.model.module, "block")
            else:
                layer = getattr(self.model, "block")
            for name, module in layer.named_modules():
                if isinstance(module, (SuperCompressAsyMobileBlockCifar)):
                    if iteration % 2 == 0:
                        bops = module.compress_fix_activation_compute_weight_bops()
                    else:
                        bops = module.compress_fix_weight_compute_activation_bops()
                    # elif iteration % 3 == 2:
                    # bops = module.compute_bops()
                    self.layer_bops[name] = bops
                    current_bops += bops

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                layer = self.model.module.conv2
            else:
                layer = self.model.conv2
            if iteration % 2 == 0:
                bops = layer.compress_fix_activation_compute_weight_bops(
                    layer.out_channels, is_out_channel=True
                )
            else:
                bops = layer.compress_fix_activation_compute_weight_bops(
                    layer.out_channels, is_out_channel=True
                )
            current_bops += bops
            self.layer_bops["conv2"] = bops

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                layer = self.model.module.conv3
            else:
                layer = self.model.conv3
            if iteration % 2 == 0:
                bops = layer.compress_fix_activation_compute_weight_bops(
                    layer.out_channels, is_out_channel=True
                )
            else:
                bops = layer.compress_fix_weight_compute_activation_bops(
                    layer.out_channels, is_out_channel=True
                )
            current_bops += bops
            self.layer_bops["conv3"] = bops

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                layer = self.model.module.fc
            else:
                layer = self.model.fc
            bops = compute_bops(1, layer.in_features, layer.out_features, 1, 1, 8, 8)
            current_bops += bops
            self.layer_bops["fc"] = bops

        bops_loss = torch.log(current_bops) * self.settings.loss_lambda
        ratio = current_bops / self.total_bops
        return bops_loss, current_bops, ratio

    def switch_gradient(self, iteration):
        if iteration % 2 == 0:
            for module in self.model.modules():
                if isinstance(
                    module,
                    (SuperQConv2d, SuperQLinear, SuperAsyQConv2d, SuperAsyQLinear),
                ):
                    module.activation_quantization_thresholds.requires_grad = False
                    module.weight_quantization_thresholds.requires_grad = True
        else:
            for module in self.model.modules():
                if isinstance(
                    module,
                    (SuperQConv2d, SuperQLinear, SuperAsyQConv2d, SuperAsyQLinear),
                ):
                    module.activation_quantization_thresholds.requires_grad = True
                    module.weight_quantization_thresholds.requires_grad = False

    def warmup_stop_pruning(self, iteration, epoch):
        if iteration < 2500 and epoch == 0:
            for module in self.model.modules():
                if isinstance(
                    module,
                    (
                        SuperCompressedGatePreBasicBlock,
                        SuperCompressedGateBasicBlock,
                        SuperCompressedGateBottleneck,
                        SuperCompressedMobileBottleneck,
                        SuperCompressAsyMobileBlockCifar,
                    ),
                ):
                    module.channel_thresholds.requires_grad = False
        else:
            for module in self.model.modules():
                if isinstance(
                    module,
                    (
                        SuperCompressedGatePreBasicBlock,
                        SuperCompressedGateBasicBlock,
                        SuperCompressedGateBottleneck,
                        SuperCompressedMobileBottleneck,
                        SuperCompressAsyMobileBlockCifar,
                    ),
                ):
                    module.channel_thresholds.requires_grad = True

    def clip_quantization_thresholds(self):
        for module in self.model.modules():
            if isinstance(
                module,
                (
                    SuperQConv2d,
                    SuperQLinear,
                    SuperQWeightConv2d,
                    SuperAsyQConv2d,
                    SuperAsyQLinear,
                ),
            ):
                module.weight_quantization_thresholds.data.clamp_(min=0)
                if hasattr(module, "activation_quantization_thresholds"):
                    module.activation_quantization_thresholds.data.clamp_(min=0)
                # module.weight_clip_value.data.clamp_(min=0)
                # module.activation_quantization_thresholds.data.clamp_(min=0)
                # module.activation_clip_value.data.clamp_(min=0)

    def clip_pruning_thresholds(self):
        # add maximum pruning rate bound
        for module in self.model.modules():
            if isinstance(
                module,
                (SuperCompressedGatePreBasicBlock, SuperCompressedGateBasicBlock,),
            ):
                filter_weight = module.conv1.weight
                # quantized_weight = normalize_and_quantize_weight(filter_weight, module.conv1.bits_weights, module.conv1.weight_clip_value.detach())
                normalized_filter_weight_norm = compute_norm(
                    filter_weight, module.group_size
                ).detach()
                max_pruning_filter_num = max(
                    int(
                        module.conv1.out_channels
                        * self.settings.max_pruning_ratio
                        / module.group_size
                    ),
                    1,
                )
                value, _ = torch.topk(
                    normalized_filter_weight_norm,
                    k=max_pruning_filter_num,
                    largest=False,
                )
                max_threshold = value[-1]
                module.channel_thresholds.data.clamp_(max=max_threshold, min=0)
            elif isinstance(module, SuperCompressedGateBottleneck):
                filter_weight = module.conv1.weight
                # quantized_weight = normalize_and_quantize_weight(filter_weight, module.conv1.bits_weights, module.conv1.weight_clip_value.detach())
                normalized_filter_weight_norm = compute_norm(
                    filter_weight, module.group_size
                ).detach()
                max_pruning_filter_num = max(
                    int(
                        module.conv1.out_channels
                        * self.settings.max_pruning_ratio
                        / module.group_size
                    ),
                    1,
                )
                value, _ = torch.topk(
                    normalized_filter_weight_norm,
                    k=max_pruning_filter_num,
                    largest=False,
                )
                max_threshold = value[-1]
                module.channel_thresholds_1.data.clamp_(max=max_threshold, min=0)

                filter_weight = module.conv2.weight
                # quantized_weight = normalize_and_quantize_weight(filter_weight, module.conv1.bits_weights, module.conv1.weight_clip_value.detach())
                normalized_filter_weight_norm = compute_norm(
                    filter_weight, module.group_size
                ).detach()
                max_pruning_filter_num = max(
                    int(
                        module.conv2.out_channels
                        * self.settings.max_pruning_ratio
                        / module.group_size
                    ),
                    1,
                )
                value, _ = torch.topk(
                    normalized_filter_weight_norm,
                    k=max_pruning_filter_num,
                    largest=False,
                )
                max_threshold = value[-1]
                module.channel_thresholds_2.data.clamp_(max=max_threshold, min=0)

            elif isinstance(
                module,
                (SuperCompressedMobileBottleneck, SuperCompressAsyMobileBlockCifar),
            ):
                filter_weight = module.conv2.weight
                normalized_filter_weight_norm = compute_norm(
                    filter_weight, module.group_size
                ).detach()
                max_pruning_filter_num = max(
                    int(
                        module.conv2.out_channels
                        * self.settings.max_pruning_ratio
                        / module.group_size
                    ),
                    1,
                )
                value, _ = torch.topk(
                    normalized_filter_weight_norm,
                    k=max_pruning_filter_num,
                    largest=False,
                )
                max_threshold = value[-1]
                module.channel_thresholds.data.clamp_(max=max_threshold, min=0)

    def train(self, epoch):
        """
        Train one epoch for auxnet
        :param epoch: index of epoch
        """

        metric_logger = MetricLogger(logger=self.logger, delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

        self.model.train()

        header = "Epoch: [{}]".format(epoch)
        i = 0
        for image, target in metric_logger.log_every(
            self.train_loader, self.settings.print_frequency, header
        ):
            start_time = time.time()
            image, target = image.to(self.device), target.to(self.device)

            self.switch_gradient(iteration=i)
            # self.warmup_stop_pruning(iteration=i, epoch=epoch)

            # forward
            output = self.model(image)
            loss = self.criterion(output, target)
            bops_loss, bops, ratio = self.compute_bops_loss(iteration=i)
            total_loss = loss + bops_loss

            # train network parameters
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.clip_quantization_thresholds()
            self.clip_pruning_thresholds()

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(
                loss=loss.item(),
                lr=self.optimizer.param_groups[0]["lr"],
                total_loss=total_loss.item(),
                bops_loss=bops_loss.item(),
                bops=bops / 1e6,
                ratio=ratio,
            )
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["img/s"].update(
                batch_size / (time.time() - start_time)
            )

            i += 1
            if i % 500 == 0:
                self.get_bits()
                self.get_channels()
                if self.settings.rank == 0:
                    if self.settings.distributed:
                        model = self.model_without_ddp
                    else:
                        model = self.model
                    self.iteration_checkpoint.save_checkpoint(
                        model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        epoch * len(self.train_loader) + i,
                    )

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        self.scheduler.step()
        self.get_lr()
        self.logger.info("Change Learning rate: {}".format(self.lr))

        train_error = 100 - metric_logger.acc1.global_avg
        train_loss = metric_logger.loss.global_avg
        train5_error = 100 - metric_logger.acc5.global_avg
        total_loss_ = metric_logger.total_loss.global_avg

        bops_ = metric_logger.bops.global_avg
        bops_loss_ = metric_logger.bops_loss.global_avg
        ratio_ = metric_logger.ratio.global_avg
        if self.tensorboard_logger is not None:
            for name, module in self.model.named_modules():
                if isinstance(
                    module,
                    (
                        SuperCompressedGatePreBasicBlock,
                        SuperCompressedGateBasicBlock,
                        SuperCompressedMobileBottleneck,
                        SuperCompressAsyMobileBlockCifar,
                    ),
                ):
                    self.tensorboard_logger.add_scalar(
                        "{}_channel_thresholds".format(name),
                        module.channel_thresholds,
                        self.run_count,
                    )
                    self.logger.info("Threshold")
                    self.logger.info(module.channel_thresholds)
                    if hasattr(module, "conv1"):
                        filter_weight = module.conv1.weight
                    else:
                        filter_weight = module.conv2.weight
                    normalized_filter_weight_norm = compute_norm(
                        filter_weight, module.group_size
                    )
                    self.logger.info("Norm")
                    self.logger.info(normalized_filter_weight_norm)
                    # quantized_weight = normalize_and_quantize_weight(filter_weight, module.conv1.bits_weights, module.conv1.weight_clip_value.data)
                    # normalized_filter_weight_norm = module.compute_norm(quantized_weight).data
                    # self.logger.info('Quantized Norm')
                    # self.logger.info(normalized_filter_weight_norm)
                    self.tensorboard_logger.add_histogram(
                        "{}_filters_norm".format(name),
                        normalized_filter_weight_norm,
                        self.run_count,
                    )
                elif isinstance(module, SuperCompressedGateBottleneck):
                    self.tensorboard_logger.add_scalar(
                        "{}_channel_thresholds_1".format(name),
                        module.channel_thresholds_1,
                        self.run_count,
                    )
                    self.logger.info("Threshold_1")
                    self.logger.info(module.channel_thresholds_1)

                    self.tensorboard_logger.add_scalar(
                        "{}_channel_thresholds_2".format(name),
                        module.channel_thresholds_2,
                        self.run_count,
                    )
                    self.logger.info("Threshold_2")
                    self.logger.info(module.channel_thresholds_2)

                    filter_weight = module.conv1.weight
                    normalized_filter_weight_norm = compute_norm(
                        filter_weight, module.group_size
                    )
                    self.logger.info("Norm_1")
                    self.logger.info(normalized_filter_weight_norm)
                    self.tensorboard_logger.add_histogram(
                        "{}_filters_norm_1".format(name),
                        normalized_filter_weight_norm,
                        self.run_count,
                    )

                    filter_weight = module.conv2.weight
                    normalized_filter_weight_norm = compute_norm(
                        filter_weight, module.group_size
                    )
                    self.logger.info("Norm_2")
                    self.logger.info(normalized_filter_weight_norm)
                    self.tensorboard_logger.add_histogram(
                        "{}_filters_norm_2".format(name),
                        normalized_filter_weight_norm,
                        self.run_count,
                    )
                elif isinstance(
                    module,
                    (
                        SuperQConv2d,
                        SuperQLinear,
                        SuperQWeightConv2d,
                        SuperAsyQConv2d,
                        SuperAsyQLinear,
                    ),
                ):
                    self.tensorboard_logger.add_scalar(
                        "{}_{}".format(name, "weight_clip_value"),
                        module.weight_clip_value,
                        self.run_count,
                    )
                    self.tensorboard_logger.add_scalar(
                        "{}_{}".format(name, "activation_clip_value"),
                        module.activation_clip_value,
                        self.run_count,
                    )
                    for i in range(len(module.weight_quantization_thresholds)):
                        self.tensorboard_logger.add_scalar(
                            "{}_{}".format(name, "weight_indicator_{}".format(i)),
                            module.weight_indicator_list[i],
                            self.run_count,
                        )
                    for i in range(len(module.weight_quantization_thresholds)):
                        self.tensorboard_logger.add_scalar(
                            "{}_{}".format(name, "weight_threshold_{}".format(i)),
                            module.weight_quantization_thresholds[i],
                            self.run_count,
                        )
                    if hasattr(module, "activation_quantization_thresholds"):
                        for i in range(
                            len(module.activation_quantization_thresholds) - 1
                        ):
                            self.tensorboard_logger.add_scalar(
                                "{}_{}".format(
                                    name, "activation_indicator_{}".format(i)
                                ),
                                module.input_indicator_list[i],
                                self.run_count,
                            )
                        for i in range(len(module.activation_quantization_thresholds)):
                            self.tensorboard_logger.add_scalar(
                                "{}_{}".format(
                                    name, "activation_threshold_{}".format(i)
                                ),
                                module.activation_quantization_thresholds[i],
                                self.run_count,
                            )

                    weight_bit = module.bits_weights_list[-1]
                    if module.weight_indicator_list[0] == 0.0:
                        weight_bit = module.bits_weights_list[0]
                    else:
                        if len(module.weight_indicator_list) == 1:
                            if module.weight_indicator_list[0] == 1.0:
                                weight_bit = module.bits_weights_list[1]
                        else:
                            if (
                                module.weight_indicator_list[0] == 1.0
                                and module.weight_indicator_list[1] == 0.0
                            ):
                                weight_bit = module.bits_weights_list[1]
                            elif (
                                module.weight_indicator_list[0] == 1.0
                                and module.weight_indicator_list[1] == 1.0
                            ):
                                weight_bit = module.bits_weights_list[2]

                    self.tensorboard_logger.add_scalar(
                        "{}_weight_bit".format(name), weight_bit, self.run_count,
                    )

                    activation_bit = module.bits_activations_list[-1]
                    if hasattr(module, "activation_quantization_thresholds"):
                        if module.input_indicator_list[0] == 0:
                            activation_bit = module.bits_activations_list[0]
                        else:
                            if len(module.input_indicator_list) == 1:
                                if module.input_indicator_list[0] == 1:
                                    activation_bit = module.bits_activations_list[1]
                            else:
                                if (
                                    module.input_indicator_list[0] == 1
                                    and module.input_indicator_list[1] == 0
                                ):
                                    activation_bit = module.bits_activations_list[1]
                                elif (
                                    module.input_indicator_list[0] == 1
                                    and module.input_indicator_list[1] == 1
                                ):
                                    activation_bit = module.bits_activations_list[2]

                    self.tensorboard_logger.add_scalar(
                        "{}_activation_bit".format(name),
                        activation_bit,
                        self.run_count,
                    )

                elif isinstance(module, (QConv2d, QLinear)):
                    self.tensorboard_logger.add_scalar(
                        "{}_{}".format(name, "weight_clip_value"),
                        module.weight_clip_value,
                        self.run_count,
                    )
                    self.tensorboard_logger.add_scalar(
                        "{}_{}".format(name, "activation_clip_value"),
                        module.activation_clip_value,
                        self.run_count,
                    )

            self.tensorboard_logger.add_scalar(
                "train_top1_error", train_error, self.run_count
            )
            self.tensorboard_logger.add_scalar(
                "train_top5_error", train5_error, self.run_count
            )
            self.tensorboard_logger.add_scalar("train_loss", train_loss, self.run_count)
            self.tensorboard_logger.add_scalar(
                "total_loss", total_loss_, self.run_count
            )
            self.tensorboard_logger.add_scalar("bops", bops_, self.run_count)
            self.tensorboard_logger.add_scalar("ratio", ratio_, self.run_count)
            self.tensorboard_logger.add_scalar("bops_loss", bops_loss_, self.run_count)
            self.tensorboard_logger.add_scalar("lr", self.lr, self.run_count)

        self.logger.info(
            "|===>Training Error: {:.4f} Loss: {:.4f}, Top5 Error: {:.4f}".format(
                train_error, train_loss, train5_error
            )
        )

        return train_error, train_loss, train5_error

    def val(self, epoch):
        """
        Validation
        :param epoch: index of epoch
        """

        self.model.eval()
        metric_logger = MetricLogger(logger=self.logger, delimiter="  ")
        header = "Test:"

        with torch.no_grad():
            for image, target in metric_logger.log_every(
                self.val_loader, self.settings.print_frequency, header
            ):
                image = image.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # forward
                output = self.model(image)
                loss = self.criterion(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                batch_size = image.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        val_error = 100 - metric_logger.acc1.global_avg
        val_loss = metric_logger.loss.global_avg
        val5_error = 100 - metric_logger.acc5.global_avg
        if self.tensorboard_logger is not None:
            for name, module in self.model.named_modules():
                if isinstance(
                    module,
                    (
                        SuperCompressedGatePreBasicBlock,
                        SuperCompressedGateBasicBlock,
                        SuperCompressedMobileBottleneck,
                        SuperCompressAsyMobileBlockCifar,
                    ),
                ):
                    num_group = module.indicator.sum().item()
                    channel_num = num_group * module.group_size

                    self.tensorboard_logger.add_scalar(
                        "{}_num_filters".format(name), channel_num, self.run_count,
                    )
                elif isinstance(module, SuperCompressedGateBottleneck):
                    num_group = module.indicator_1.sum().item()
                    channel_num = num_group * module.group_size

                    self.tensorboard_logger.add_scalar(
                        "{}_num_filters_1".format(name), channel_num, self.run_count,
                    )

                    num_group = module.indicator_2.sum().item()
                    channel_num = num_group * module.group_size

                    self.tensorboard_logger.add_scalar(
                        "{}_num_filters_2".format(name), channel_num, self.run_count,
                    )
            self.tensorboard_logger.add_scalar(
                "val_top1_error", val_error, self.run_count
            )
            self.tensorboard_logger.add_scalar(
                "val_top5_error", val5_error, self.run_count
            )
            self.tensorboard_logger.add_scalar("val_loss", val_loss, self.run_count)

        self.run_count += 1
        self.logger.info(
            "|===>Testing Error: {:.4f} Loss: {:.4f}, Top5 Error: {:.4f}".format(
                val_error, val_loss, val5_error
            )
        )
        return val_error, val_loss, val5_error

    def get_bits(self):
        for name, module in self.model.named_modules():
            if isinstance(
                module,
                (
                    SuperQConv2d,
                    SuperQLinear,
                    SuperQWeightConv2d,
                    SuperAsyQConv2d,
                    SuperAsyQLinear,
                ),
            ):
                weight_indicator_list = module.weight_indicator_list
                if weight_indicator_list[0] == 0:
                    module.bits_weights.data.fill_(module.bits_weights_list[0])
                else:
                    if len(weight_indicator_list) == 1:
                        if weight_indicator_list[0] == 1:
                            module.bits_weights.data.fill_(module.bits_weights_list[1])
                    else:
                        if (
                            weight_indicator_list[0] == 1
                            and weight_indicator_list[1] == 0
                        ):
                            module.bits_weights.data.fill_(module.bits_weights_list[1])
                        elif (
                            weight_indicator_list[0] == 1
                            and weight_indicator_list[1] == 1
                        ):
                            module.bits_weights.data.fill_(module.bits_weights_list[2])

                if hasattr(module, "activation_quantization_thresholds"):
                    activation_indicator_list = module.input_indicator_list
                    if activation_indicator_list[0] == 0:
                        module.bits_activations.data.fill_(
                            module.bits_activations_list[0]
                        )
                    else:
                        if len(activation_indicator_list) == 1:
                            if activation_indicator_list[0] == 1:
                                module.bits_activations.data.fill_(
                                    module.bits_activations_list[1]
                                )
                        else:
                            if (
                                activation_indicator_list[0] == 1
                                and activation_indicator_list[1] == 0
                            ):
                                module.bits_activations.data.fill_(
                                    module.bits_activations_list[1]
                                )
                            elif (
                                activation_indicator_list[0] == 1
                                and activation_indicator_list[1] == 1
                            ):
                                module.bits_activations.data.fill_(
                                    module.bits_activations_list[2]
                                )
                self.logger.info(
                    "Layer: {}, bit_W: {}, bit_A: {}".format(
                        name, module.bits_weights.item(), module.bits_activations.item()
                    )
                )

    def get_channels(self):
        for name, module in self.model.named_modules():
            if isinstance(
                module,
                (
                    SuperCompressedGatePreBasicBlock,
                    SuperCompressedGateBasicBlock,
                    SuperCompressedGateBasicBlockGroup,
                    SuperCompressedMobileBottleneck,
                    SuperCompressAsyMobileBlockCifar,
                ),
            ):
                module.compute_indicator()
                num_group = module.indicator.sum().item()
                channel_num = num_group * module.group_size
                self.logger.info("Layer: {}, channel: {}".format(name, channel_num))
            elif isinstance(
                module,
                (SuperCompressedGateBottleneck, SuperCompressedGateBottleneckGroup),
            ):
                module.compute_indicator()
                num_group_1 = module.indicator_1.sum().item()
                channel_num_1 = num_group_1 * module.group_size

                num_group_2 = module.indicator_1.sum().item()
                channel_num_2 = num_group_2 * module.group_size
                self.logger.info(
                    "Layer: {}, channel: {}, {}".format(
                        name, channel_num_1, channel_num_2
                    )
                )

        _, current_bops, ratio = self.compute_bops_loss(0)
        self.logger.info(
            "Current bops: {}M, ratio: {}".format(current_bops / 1e6, ratio)
        )

