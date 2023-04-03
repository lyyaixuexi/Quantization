import copy

import torch
from torch import distributed, nn

from compression.utils.utils import Dict


class SuperConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        out_channels_list=[],
        kernel_size=None,
        kernel_size_list=[],
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size
        self.kernel_size_list = kernel_size_list
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        max_out_channels = max(out_channels_list) if out_channels_list else out_channels
        max_kernel_size = max(kernel_size_list) if kernel_size_list else kernel_size

        channel_masks = []
        prev_out_channels = None
        for out_channels in out_channels_list:
            channel_mask = torch.ones(max_out_channels)
            channel_mask *= nn.functional.pad(
                torch.ones(out_channels), [0, max_out_channels - out_channels], value=0
            )
            if prev_out_channels:
                channel_mask *= nn.functional.pad(
                    torch.zeros(prev_out_channels),
                    [0, max_out_channels - prev_out_channels],
                    value=1,
                )
            channel_mask = channel_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            prev_out_channels = out_channels
            channel_masks.append(channel_mask)

        self.register_buffer(
            "channel_masks",
            torch.stack(channel_masks, dim=0) if out_channels_list else None,
        )
        # self.register_parameter(
        #     "channel_thresholds",
        #     nn.Parameter(torch.zeros(len(out_channels_list)))
        #     if out_channels_list
        #     else None,
        # )
        self.register_buffer(
            "channel_thresholds",
            nn.Parameter(torch.zeros(len(out_channels_list)))
            if out_channels_list
            else None,
        )

        kernel_masks = []
        prev_kernel_size = None
        for kernel_size in kernel_size_list:
            kernel_mask = torch.ones(max_kernel_size, max_kernel_size)
            kernel_mask *= nn.functional.pad(
                torch.ones(kernel_size, kernel_size),
                [(max_kernel_size - kernel_size) // 2] * 4,
                value=0,
            )
            if prev_kernel_size:
                kernel_mask *= nn.functional.pad(
                    torch.zeros(prev_kernel_size, prev_kernel_size),
                    [(max_kernel_size - prev_kernel_size) // 2] * 4,
                    value=1,
                )
            kernel_mask = kernel_mask.unsqueeze(0).unsqueeze(0)
            prev_kernel_size = kernel_size
            kernel_masks.append(kernel_mask)

        self.register_buffer(
            "kernel_masks",
            torch.stack(kernel_masks, dim=0) if kernel_size_list else None,
        )
        # self.register_parameter(
        #     "kernel_thresholds",
        #     nn.Parameter(torch.zeros(len(kernel_size_list)))
        #     if kernel_size_list
        #     else None,
        # )
        self.register_buffer(
            "kernel_thresholds",
            nn.Parameter(torch.zeros(len(kernel_size_list)))
            if kernel_size_list
            else None,
        )

        self.register_parameter(
            "weight",
            nn.Parameter(
                torch.Tensor(
                    max_out_channels,
                    in_channels // groups,
                    max_kernel_size,
                    max_kernel_size,
                )
            ),
        )
        self.register_parameter(
            "bias", nn.Parameter(torch.Tensor(max_out_channels)) if bias else None
        )

        self.max_out_channels = max_out_channels
        self.max_kernel_size = max_kernel_size

    def forward(self, input):
        weight = self.weight
        if self.channel_masks is not None and self.channel_thresholds is not None:
            weight = weight * self.parametrized_mask(
                list(self.channel_masks), list(self.channel_thresholds)
            )
        if self.kernel_masks is not None and self.kernel_thresholds is not None:
            weight = weight * self.parametrized_mask(
                list(self.kernel_masks), list(self.kernel_thresholds)
            )
        return nn.functional.conv2d(
            input,
            weight,
            self.bias,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )

    def parametrized_mask(self, masks, thresholds):
        if not masks or not thresholds:
            return 0
        mask = masks.pop(0)
        threshold = thresholds.pop(0)
        norm = torch.norm(self.weight * mask)
        indicator = (
            (norm > threshold).float()
            - torch.sigmoid(norm - threshold).detach()
            + torch.sigmoid(norm - threshold)
        )
        return indicator * (mask + self.parametrized_mask(masks, thresholds))

    def freeze_weight(self):
        weight = self.weight
        if self.channel_masks is not None and self.channel_thresholds is not None:
            prev_out_channels = None
            for channel_mask, channel_threshold, out_channels in zip(
                self.channel_masks, self.channel_thresholds, self.out_channels_list
            ):
                if prev_out_channels:
                    channel_norm = torch.norm(self.weight * channel_mask)
                    if channel_norm < channel_threshold:
                        weight = weight[..., :prev_out_channels]
                        break
                prev_out_channels = out_channels
        if self.kernel_masks is not None and self.kernel_thresholds is not None:
            prev_kernel_size = None
            for kernel_mask, kernel_threshold, kernel_size in zip(
                self.kernel_masks, self.kernel_thresholds, self.kernel_size_list
            ):
                if prev_kernel_size:
                    kernel_norm = torch.norm(self.weight * kernel_mask)
                    if kernel_norm < kernel_threshold:
                        cut = (self.max_kernel_size - prev_kernel_size) // 2
                        weight = weight[..., cut:-cut, cut:-cut]
                        break
                prev_kernel_size = kernel_size
        self.weight = weight


class SuperMobileConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, expand_ratio_list, kernel_size_list, stride):

        super().__init__()

        hidden_channels_list = [in_channels * expand_ratio for expand_ratio in expand_ratio_list]
        max_hidden_channels = max(hidden_channels_list)
        max_kernel_size = max(kernel_size_list)

        self.module = nn.Sequential(
            nn.Sequential(
                SuperConv2d(
                    in_channels=in_channels,
                    out_channels_list=hidden_channels_list,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(max_hidden_channels),
                nn.ReLU6()
            ),
            nn.Sequential(
                SuperConv2d(
                    in_channels=max_hidden_channels,
                    out_channels=max_hidden_channels,
                    groups=max_hidden_channels,
                    kernel_size_list=kernel_size_list,
                    padding=(max_kernel_size - 1) // 2,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(max_hidden_channels),
                nn.ReLU6()
            ),
            nn.Sequential(
                nn.Conv2d(
                    in_channels=max_hidden_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            ),
        )

    def forward(self, input):
        output = self.module(input)
        if input.shape == output.shape:
            output += input
        return output


class SuperMobileNetV2(nn.Module):

    def __init__(self, first_conv_param, middle_conv_params, last_conv_param, num_classes, drop_prob=0.2):

        super().__init__()

        self.module = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=first_conv_param.in_channels,
                    out_channels=first_conv_param.out_channels,
                    kernel_size=first_conv_param.kernel_size,
                    padding=(first_conv_param.kernel_size - 1) // 2,
                    stride=first_conv_param.stride,
                    bias=False
                ),
                nn.BatchNorm2d(first_conv_param.out_channels),
                nn.ReLU6()
            ),
            nn.Sequential(*[
                nn.Sequential(*[
                    SuperMobileConvBlock(
                        in_channels=middle_conv_param.out_channels if i else middle_conv_param.in_channels,
                        out_channels=middle_conv_param.out_channels,
                        expand_ratio_list=middle_conv_param.expand_ratio_list,
                        kernel_size_list=middle_conv_param.kernel_size_list,
                        stride=1 if i else middle_conv_param.stride
                    ) for i in range(middle_conv_param.blocks)
                ]) for middle_conv_param in middle_conv_params
            ]),
            nn.Sequential(
                nn.Conv2d(
                    in_channels=last_conv_param.in_channels,
                    out_channels=last_conv_param.out_channels,
                    kernel_size=last_conv_param.kernel_size,
                    padding=(last_conv_param.kernel_size - 1) // 2,
                    stride=last_conv_param.stride,
                    bias=False
                ),
                nn.BatchNorm2d(last_conv_param.out_channels),
                nn.ReLU6()
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Dropout(drop_prob),
                nn.Conv2d(
                    in_channels=last_conv_param.out_channels,
                    out_channels=num_classes,
                    kernel_size=1,
                    bias=True
                )
            )
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, SuperConv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input):
        return self.module(input).squeeze()

    def train_thresholds(self, val_images, val_labels, threshold_optimizer, criterion, config):

        val_logits = self(val_images)
        val_loss = criterion(val_logits, val_labels) / config.world_size

        threshold_optimizer.zero_grad()

        val_loss.backward()

        for threshold in self.thresholds():
            distributed.all_reduce(threshold.grad)

        threshold_optimizer.step()

        return val_logits, val_loss

    def train_thresholds_bilevel(self, train_images, train_targets, val_images, val_targets, weight_optimizer, threshold_optimizer, criterion, config):

        # Save current network parameters and optimizer.
        named_weights = copy.deepcopy(list(self.named_weights()))
        named_buffers = copy.deepcopy(list(self.named_buffers()))
        weight_optimizer_state_dict = copy.deepcopy(weight_optimizer.state_dict())

        # Approximate w*(α) by adapting w using only a single training step,
        # without solving the inner optimization completely by training until convergence.
        train_logits = self(train_images)
        train_loss = criterion(train_logits, train_targets) / config.world_size

        weight_optimizer.zero_grad()

        train_loss.backward()

        for weight in self.weights():
            distributed.all_reduce(weight.grad)

        weight_optimizer.step()

        # Apply chain rule to the approximate architecture gradient.
        # Backward validation loss, but don't update approximate parameter w'.
        val_logits = self(val_images)
        val_loss = criterion(val_logits, val_targets) / config.world_size

        weight_optimizer.zero_grad()
        threshold_optimizer.zero_grad()

        val_loss.backward()

        named_weight_gradients = copy.deepcopy([(name, weight.grad) for name, weight in self.named_weights()])
        weight_gradient_norm = torch.norm(torch.cat([weight_gradient.reshape(-1) for name, weight_gradient in named_weight_gradients]))

        # Avoid calculate hessian-vector product using the finite difference approximation.
        for weight, (_, prev_weight), (_, prev_weight_gradient) in zip(self.weights(), named_weights, named_weight_gradients):
            weight.data = (prev_weight + prev_weight_gradient * config.epsilon / weight_gradient_norm).data

        train_logits = self(train_images)
        train_loss = criterion(train_logits, train_targets) * -(config.weight_optimizer.lr / (2 * config.epsilon / weight_gradient_norm)) / config.world_size

        train_loss.backward()

        # Avoid calculate hessian-vector product using the finite difference approximation.
        for weight, (_, prev_weight), (_, prev_weight_gradient) in zip(self.weights(), named_weights, named_weight_gradients):
            weight.data = (prev_weight - prev_weight_gradient * config.epsilon / weight_gradient_norm).data

        train_logits = self(train_images)
        train_loss = criterion(train_logits, train_targets) * (config.weight_optimizer.lr / (2 * config.epsilon / weight_gradient_norm)) / config.world_size

        train_loss.backward()

        # Finally, update architecture parameter.
        for threshold in self.thresholds():
            distributed.all_reduce(threshold.grad)

        threshold_optimizer.step()

        # Restore previous network parameters and optimizer.
        self.load_state_dict(dict(**dict(named_weights), **dict(named_buffers)), strict=True)
        weight_optimizer.load_state_dict(weight_optimizer_state_dict)

        return val_logits, val_loss

    def train_weights(self, train_images, train_targets, weight_optimizer, criterion, config):

        train_logits = self(train_images)
        train_loss = criterion(train_logits, train_targets) / config.world_size

        weight_optimizer.zero_grad()

        train_loss.backward()

        for weight in self.weights():
            distributed.all_reduce(weight.grad)

        weight_optimizer.step()

        return train_logits, train_loss

    def freeze_weight(self):
        for module in self.modules():
            if isinstance(module, SuperConv2d):
                module.freeze_weight()

    def weights(self):
        for name, parameter in self.named_parameters():
            if 'threshold' not in name:
                yield parameter

    def named_weights(self):
        for name, parameter in self.named_parameters():
            if 'threshold' not in name:
                yield name, parameter

    def weight_gradients(self):
        for name, parameter in self.named_parameters():
            if 'threshold' not in name:
                yield parameter.grad

    def named_weight_gradients(self):
        for name, parameter in self.named_parameters():
            if 'threshold' not in name:
                yield name, parameter.grad

    def thresholds(self):
        for name, parameter in self.named_parameters():
            if 'threshold' in name:
                yield parameter

    def named_thresholds(self):
        for name, parameter in self.named_parameters():
            if 'threshold' in name:
                yield name, parameter

    def threshold_gradients(self):
        for name, parameter in self.named_parameters():
            if 'threshold' in name:
                yield parameter.grad

    def named_threshold_gradients(self):
        for name, parameter in self.named_parameters():
            if 'threshold' in name:
                yield name, parameter.grad

def get_super_mobilenetv2(num_classes=1000):
    model = SuperMobileNetV2(
        first_conv_param=Dict(in_channels=3, out_channels=32, kernel_size=3, stride=2),
        middle_conv_params=[
            Dict(in_channels=32, out_channels=16, expand_ratio_list=[3, 6], kernel_size_list=[3, 5], blocks=1, stride=1),
            Dict(in_channels=16, out_channels=24, expand_ratio_list=[3, 6], kernel_size_list=[3, 5], blocks=2, stride=2),
            Dict(in_channels=24, out_channels=32, expand_ratio_list=[3, 6], kernel_size_list=[3, 5], blocks=3, stride=2),
            Dict(in_channels=32, out_channels=64, expand_ratio_list=[3, 6], kernel_size_list=[3, 5], blocks=4, stride=2),
            Dict(in_channels=64, out_channels=96, expand_ratio_list=[3, 6], kernel_size_list=[3, 5], blocks=3, stride=1),
            Dict(in_channels=96, out_channels=160, expand_ratio_list=[3, 6], kernel_size_list=[3, 5], blocks=3, stride=2),
            Dict(in_channels=160, out_channels=320, expand_ratio_list=[3, 6], kernel_size_list=[3, 5], blocks=1, stride=1),
        ],
        last_conv_param=Dict(in_channels=320, out_channels=1280, kernel_size=1, stride=1),
        drop_prob=0.2,
        num_classes=num_classes
    )
    return model
