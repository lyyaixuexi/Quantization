import math

import torch
import torch.nn as nn

from compression.utils.utils import *

from compression.models.quantization import lsq
from compression.models.quantization import dorefa_clip
from compression.models.quantization import dorefa_clip_asymmetric
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
        lr_scheduler_state=None,
        run_count=0,
    ):
        super(QTrainer, self).__init__(
            model,
            device,
            train_loader,
            val_loader,
            settings,
            logger,
            tensorboard_logger,
            optimizer_state,
            lr_scheduler_state,
            run_count,
        )
        self.layer_input = {}
        self.hooks = []

    def forward_hook(self, module, input, output):
        name = module.name
        self.layer_input[name] = input[0]

    def init_scale(self):
        for name, module in self.model.named_modules():
            if isinstance(module, lsq.QConv2d):
                module.name = name
                hook = module.register_forward_hook(self.forward_hook)
                self.hooks.append(hook)
                module.bits_weights = 32
                module.bits_activations = 32

        self.model.eval()
        with torch.no_grad():
            train_dataloader_iter = iter(self.train_loader)
            images, _ = train_dataloader_iter.next()
            self.model(images.cuda())

        for hook in self.hooks:
            hook.remove()

        for name, module in self.model.named_modules():
            if isinstance(module, lsq.QConv2d):
                module.init_activation_scale(self.layer_input[name])
                module.init_weight_scale()

        # bn_dict = {}

        # for name, module in self.model.named_modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         bn_dict[name] = module

        # for name, module in self.model.named_modules():
        #     if isinstance(module, lsq.QConv2d):
        #         bn_name = name.replace('conv', 'bn')
        #         bn_module = bn_dict[bn_name]
        #         bn_weight = bn_module.weight
        #         bn_bias = bn_module.bias
        # print('weight shape: {}'.format(bn_weight.shape))
        # print('bias shape: {}'.format(bn_bias.shape))

        # module.init_activation_scale(self.layer_input[name])
        # module.init_weight_scale()
        # assert False

    def backward(self, loss):
        """
        backward propagation
        """
        self.optimizer.zero_grad()
        loss.backward()
        for name, module in self.model.named_modules():
            if isinstance(module, (lsq.QConv2d, lsq.QLinear)):
                grad_weight_scale = math.sqrt(module.weight.nelement() * module.wqp)
                module.w_s.grad.data /= grad_weight_scale
                if module.a_s.grad is not None:
                    grad_activation_scale = math.sqrt(module.input_nelement * module.aqp)
                    module.a_s.grad.data /= grad_activation_scale
        self.optimizer.step()

    def init_weight(self):
        for name, module in self.model.named_modules():
            if isinstance(module, dorefa_clip.QConv2d):
                mean = module.weight.data.mean()
                std = module.weight.data.std()
                init_value = mean + 3 * std
                self.logger.info("mean: {}, std: {}".format(mean, std))
                self.logger.info(
                    "{} init weight clip value to {}".format(name, init_value.item())
                )
                # module.weight_clip_value.data.fill_(module.weight.data.max())
                module.weight_clip_value.data.fill_(init_value)

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
            metric_logger.update(
                loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"]
            )
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["img/s"].update(
                batch_size / (time.time() - start_time)
            )

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        self.scheduler.step()
        self.get_lr()
        self.logger.info("Change Learning rate: {}".format(self.lr))

        train_error = 100 - metric_logger.acc1.global_avg
        train_loss = metric_logger.loss.global_avg
        train5_error = 100 - metric_logger.acc5.global_avg
        if self.tensorboard_logger is not None:
            for name, module in self.model.named_modules():
                if isinstance(module, lsq.QConv2d):
                    self.tensorboard_logger.add_scalar(
                        "{}_{}".format(name, "w_s"), module.w_s, self.run_count
                    )
                    self.tensorboard_logger.add_scalar(
                        "{}_{}".format(name, "a_s"), module.a_s, self.run_count
                    )
                if isinstance(module, (dorefa_clip.QConv2d, dorefa_clip.QLinear, dorefa_clip_asymmetric.QConv2d, dorefa_clip_asymmetric.QLinear)):
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

                    if isinstance(module, (dorefa_clip_asymmetric.QConv2d, dorefa_clip_asymmetric.QLinear)):
                        self.tensorboard_logger.add_scalar(
                            "{}_{}".format(name, "activation_bias"),
                            module.activation_bias,
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
