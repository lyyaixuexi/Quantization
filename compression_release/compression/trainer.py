import time

import torch.autograd
import torch.nn as nn

from compression.models.quantization.dorefa_clip import QConv2d, QLinear
from compression.utils.label_smooth import LabelSmoothCrossEntropyLoss
from compression.utils.scheduler import GradualWarmupScheduler
from compression.utils.utils import *


class View(nn.Module):
    """
    Reshape data from 4 dimension to 2 dimension
    """

    def forward(self, x):
        assert x.dim() == 2 or x.dim() == 4, "invalid dimension of input {:d}".format(x.dim())
        if x.dim() == 4:
            out = x.view(x.size(0), -1)
        else:
            out = x
        return out


class Trainer(object):
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

        if "mobilenetv3" in self.settings.net_type:
            self.criterion = LabelSmoothCrossEntropyLoss(num_classes=self.settings.n_classes)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.lr = self.settings.lr
        if "mobilenetv3" in self.settings.net_type and "imagenet" in self.settings.dataset:
            no_decay_keys = ["bn", "bias"]
            params = []
            for name, param in self.model.named_parameters():
                flag = False
                for key in no_decay_keys:
                    if key in name:
                        flag = True
                        break
                if flag:
                    weight_decay = 0
                else:
                    weight_decay = self.settings.weight_decay
                params.append(
                    {"params": param, "lr": self.settings.lr, "weight_decay": weight_decay}
                )
            if self.settings.opt_type == "SGD":
                self.optimizer = torch.optim.SGD(
                    params=params, momentum=self.settings.momentum, nesterov=True
                )
        elif "mobilenet" in self.settings.net_type and "mobilenet_imagenet" in self.settings.net_type:
            params = []
            for param in self.model.parameters():
                param_size = list(param.size())
                # do not put weight decay on depth-wise convolution
                if len(param_size) == 4 and param_size[1] != 1:
                    weight_decay = self.settings.weight_decay
                elif len(param_size) == 2:
                    weight_decay = self.settings.weight_decay
                else:
                    weight_decay = 0
                params.append(
                    {"params": param, "lr": self.settings.lr, "weight_decay": weight_decay}
                )
            if self.settings.opt_type == "SGD":
                self.optimizer = torch.optim.SGD(
                    params=params, momentum=self.settings.momentum, nesterov=True
                )
        else:
            if "SGD" in self.settings.opt_type:
                self.optimizer = torch.optim.SGD(
                    params=self.model.parameters(),
                    lr=self.settings.lr,
                    momentum=self.settings.momentum,
                    weight_decay=self.settings.weight_decay,
                    nesterov=True,
                )
            elif "RMSProp" in self.settings.opt_type:
                self.optimizer = torch.optim.RMSprop(
                    params=self.model.parameters(),
                    lr=self.settings.lr,
                    alpha=self.settings.alpha,
                    eps=self.settings.eps,
                    weight_decay=self.settings.weight_decay,
                    momentum=self.settings.momentum
                )

        self.logger.info("mobilenet" in self.settings.net_type)
        self.logger.info(self.optimizer)
        self.logger.info(self.criterion)
        # assert False

        if optimizer_state is not None:
            self.logger.info("Load optimizer state!")
            self.optimizer.load_state_dict(optimizer_state)

        if "cosine_warmup" in self.settings.lr_scheduler_type:
            self.logger.info("Cosine Annealing Warmup LR!")
            self.after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.settings.n_epochs - self.settings.warmup_n_epochs
            )
            self.scheduler = GradualWarmupScheduler(self.optimizer, 1, self.settings.warmup_n_epochs, self.after_scheduler)
        elif "cosine" in self.settings.lr_scheduler_type:
            self.logger.info("Cosine Annealing LR!")
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.settings.n_epochs
            )
        elif "multi_step_warmup" in self.settings.lr_scheduler_type:
            self.logger.info("MultiStep Warmup LR!")
            self.after_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.settings.step, gamma=0.1
            )
            self.scheduler = GradualWarmupScheduler(self.optimizer, 1, self.settings.warmup_n_epochs, self.after_scheduler)
        else:
            self.logger.info("MultiStep LR!")
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.settings.step, gamma=0.1
            )
        # if "mobilenet" in self.settings.net_type:
        #     self.logger.info("Cosine Annealing LR!")
        #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #         self.optimizer, self.settings.n_epochs
        #     )
        # else:
        #     self.logger.info("MultiStep LR!")
        #     self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #         self.optimizer, milestones=self.settings.step, gamma=0.1
        #     )

        if lr_scheduler_state is not None:
            self.logger.info("Load lr state")
            last_epoch = lr_scheduler_state["last_epoch"]
            self.logger.info(self.scheduler.last_epoch)
            while self.scheduler.last_epoch < last_epoch:
                self.scheduler.step()
            # self.scheduler.load_state_dict(lr_scheduler_state)
            # self.logger.info(lr_scheduler_state)
            # assert False

        model_without_ddp = self.model
        if self.settings.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.settings.gpu],
                find_unused_parameters=True
            )
            self.model_without_ddp = self.model.module

    def backward(self, loss):
        """
        backward propagation
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            self.lr = param_group["lr"]
            break

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
        for image, target in metric_logger.log_every(
            self.train_loader, self.settings.print_frequency, header
        ):
            start_time = time.time()
            image, target = image.to(self.device), target.to(self.device)

            # forward
            output = self.model(image)
            loss = self.criterion(output, target)

            # backward
            self.backward(loss)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        self.scheduler.step()
        self.get_lr()
        self.logger.info("Change Learning rate: {}".format(self.lr))

        train_error = 100 - metric_logger.acc1.global_avg
        train_loss = metric_logger.loss.global_avg
        train5_error = 100 - metric_logger.acc5.global_avg
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.add_scalar("train_top1_error", train_error, self.run_count)
            self.tensorboard_logger.add_scalar("train_top5_error", train5_error, self.run_count)
            self.tensorboard_logger.add_scalar("train_loss", train_loss, self.run_count)
            self.tensorboard_logger.add_scalar("lr", self.lr, self.run_count)
            for name, module in self.model.named_modules():
                if isinstance(module, (QConv2d, QLinear)):
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

    def val_without_tb(self, epoch):
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
        self.logger.info(
            "|===>Testing Error: {:.4f} Loss: {:.4f}, Top5 Error: {:.4f}".format(
                val_error, val_loss, val5_error
            )
        )
        return val_error, val_loss, val5_error
