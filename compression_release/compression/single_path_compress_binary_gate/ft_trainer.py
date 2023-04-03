import math

import torch
import torch.nn as nn

import compression.utils as utils
from compression.utils.utils import *

from compression.models.quantization.super_prune_preresnet import SuperPrunedPreBasicBlock
from compression.trainer import Trainer


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
        arch_optimizer_state=None,
        lr_scheduler_state=None,
        arch_lr_scheduler_state=None,
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
        self._valid_iter = None

        self.criterion = nn.CrossEntropyLoss()
        self.lr = self.settings.lr
        network_params = []
        arch_params = []
        for name, param in self.model.named_parameters():
            weight_decay = self.settings.weight_decay
            if "choices_params" in name:
                arch_params.append(
                    {"params": param, "lr": self.settings.arch_lr, "weight_decay": self.settings.arch_weight_decay}
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
            # self.arch_optimizer = torch.optim.SGD(
            #     params=arch_params,
            #     lr=self.settings.arch_lr,
            #     momentum=self.settings.momentum,
            #     weight_decay=self.settings.weight_decay,
            #     nesterov=True,
            # )
            self.arch_optimizer = torch.optim.Adam(
                params=arch_params,
                lr=self.settings.arch_lr,
                weight_decay=self.settings.arch_weight_decay,
                betas=(0.5, 0.999)
            )

        self.logger.info(self.optimizer)
        self.logger.info(self.arch_optimizer)
        # assert False

        if optimizer_state is not None:
            self.logger.info("Load optimizer state!")
            self.optimizer.load_state_dict(optimizer_state)

        if arch_optimizer_state is not None:
            self.logger.info("Load arch optimizer state!")
            self.arch_optimizer.load_state_dict(arch_optimizer_state)

        if "cosine" in self.settings.lr_scheduler_type:
            self.logger.info("Cosine Annealing LR!")
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.settings.n_epochs
            )
            self.arch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.arch_optimizer, self.settings.n_epochs
            )
        else:
            self.logger.info("MultiStep LR!")
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.settings.step, gamma=0.1
            )
            self.arch_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.arch_optimizer, milestones=self.settings.step, gamma=0.1
            )

        if lr_scheduler_state is not None:
            self.logger.info("Load lr state")
            last_epoch = lr_scheduler_state["last_epoch"]
            self.logger.info(self.scheduler.last_epoch)
            while self.scheduler.last_epoch < last_epoch:
                self.scheduler.step()
        
        if arch_lr_scheduler_state is not None:
            self.logger.info("Load arch lr state")
            last_epoch = arch_lr_scheduler_state["last_epoch"]
            self.logger.info(self.arch_scheduler.last_epoch)
            while self.arch_scheduler.last_epoch < last_epoch:
                self.arch_scheduler.step()

        model_without_ddp = self.model
        if self.settings.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.settings.gpu]
            )
            self.model_without_ddp = self.model.module
        
        self.hook_list = []

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                self.hook_list.append(module.register_forward_hook(self.forward_hook))

        if "cifar" in self.settings.dataset:
            random_input = torch.randn(1,3,32,32).cuda()
        else:
            random_input = torch.randn(1,3,224,224).cuda()
        self.model(random_input)

        self.total_bops = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                bops = self.compute_bops(module.kernel_size[0], module.in_channels, module.out_channels, module.h, module.w)
                self.total_bops += bops
                self.logger.info('Layer: {}, Bops: {}M'.format(name, bops/1e6))
            
            if isinstance(module, nn.Linear):
                bops = self.compute_bops(1, module.in_features, module.out_features, 1, 1)
                self.total_bops += bops
                self.logger.info('Layer: {}, Bops: {}M'.format(name, bops/1e6))
        
        for hook in self.hook_list:
            hook.remove()

        self.logger.info('Total BOPs: {}M'.format(self.total_bops/1e6))
        self.layer_bops = {}

    def forward_hook(self, module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        _, _, w, h = output.shape
        module.w = w
        module.h = h

    def compute_bops(self, kernel_size, in_channels, out_channels, h, w, bits_w=32, bits_a=32):
        nk_square = in_channels * kernel_size * kernel_size
        bop = (
            out_channels
            * nk_square
            * (
                bits_w * bits_a
                + bits_w
                + bits_a
                + math.log(nk_square, 2)
            )
            * h
            * w
        )
        return bop

    def compute_bops_loss(self, epoch):
        current_bops = 0

        if self.settings.net_type == 'super_prune_preresnet':
            layer = self.model.conv
            bops = self.compute_bops(layer.kernel_size[0], layer.in_channels, layer.out_channels, layer.h, layer.w)
            current_bops += bops
            self.layer_bops['conv'] = bops

            for i in range(3):
                layer = getattr(self.model, "layer{}".format(i + 1))
                for name, module in layer.named_modules():
                    if isinstance(module, SuperPrunedPreBasicBlock):
                        bops = module.get_bops()
                        self.layer_bops[name] = bops
                        current_bops += bops
            
            layer = self.model.fc
            bops = self.compute_bops(1, layer.in_features, layer.out_features, 1, 1)
            current_bops += bops
            self.layer_bops['fc'] = bops
        
        # init_lambda = 0.1
        # current_lambda = init_lambda + epoch / 50.0 * 4
        bops_loss = torch.log(current_bops) * self.settings.loss_lambda
        ratio = current_bops / self.total_bops
        # return loss_factor, total_bops, ratio
        return bops_loss, current_bops, ratio

    def compute_entropy(self):
        total_entropy = 0
        for module in self.model.modules():
            if isinstance(module, SuperPrunedPreBasicBlock):
                entropy = module.entropy()
                total_entropy += entropy
        return total_entropy

    @property
    def valid_next_batch(self):
        if self._valid_iter is None:
            self._valid_iter = iter(self.val_loader)
        try:
            data = next(self._valid_iter)
        except StopIteration:
            self._valid_iter = iter(self.val_loader)
            data = next(self._valid_iter)
        return data

    def train(self, epoch):
        """
        Train one epoch for auxnet
        :param epoch: index of epoch
        """

        metric_logger = MetricLogger(logger=self.logger, delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

        arch_metric_logger = MetricLogger(logger=self.logger, delimiter="  ")

        self.model.train()

        header = "Epoch: [{}]".format(epoch)
        i = 0
        for image, target in metric_logger.log_every(
            self.train_loader, self.settings.print_frequency, header
        ):
            start_time = time.time()
            image, target = image.to(self.device), target.to(self.device)

            output = self.model(image)
            loss = self.criterion(output, target)

            # train network parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(
                loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"], 
                # loss_factor=loss_factor
            )
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["img/s"].update(
                batch_size / (time.time() - start_time)
            )
            i += 1

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        self.scheduler.step()
        self.arch_scheduler.step()
        self.get_lr()
        self.logger.info("Change Learning rate: {}".format(self.lr))

        train_error = 100 - metric_logger.acc1.global_avg
        train_loss = metric_logger.loss.global_avg
        train5_error = 100 - metric_logger.acc5.global_avg
        if self.tensorboard_logger is not None:
            for name, module in self.model.named_modules():
                if isinstance(module, (SuperPrunedPreBasicBlock)):
                    for index_i in range(len(module.choices_params)):
                        self.tensorboard_logger.add_scalar(
                            "{}_{}_{}".format(name, "choices_params", index_i),
                            module.choices_params[index_i],
                            self.run_count,
                        )
                    

            self.tensorboard_logger.add_scalar(
                "train_top1_error", train_error, self.run_count
            )
            self.tensorboard_logger.add_scalar(
                "train_top5_error", train5_error, self.run_count
            )
            self.tensorboard_logger.add_scalar("train_loss", train_loss, self.run_count)
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

        for module in self.model.modules():
            if isinstance(module, SuperPrunedPreBasicBlock):
                module.set_chosen_op_active()

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
                if isinstance(module, (SuperPrunedPreBasicBlock)):
                    active_index = module.active_index[0]
                    channel_num = module.out_channels_list[active_index]

                    self.tensorboard_logger.add_scalar(
                        "{}_num_filters".format(name),
                        channel_num,
                        self.run_count,
                    )
            self.tensorboard_logger.add_scalar("val_top1_error", val_error, self.run_count)
            self.tensorboard_logger.add_scalar("val_top5_error", val5_error, self.run_count)
            self.tensorboard_logger.add_scalar("val_loss", val_loss, self.run_count)

        self.run_count += 1
        self.logger.info(
            "|===>Testing Error: {:.4f} Loss: {:.4f}, Top5 Error: {:.4f}".format(
                val_error, val_loss, val5_error
            )
        )
        return val_error, val_loss, val5_error
    
    def get_channels(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (SuperPrunedPreBasicBlock)):
                active_index = module.active_index[0]
                channel_num = module.out_channels_list[active_index]
                self.logger.info('Layer: {}, channel: {}'.format(name, channel_num))
