import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
import torch.distributed as dist
from compression.utils.utils import *


def quantization(x, k):
    n = 2 ** k - 1
    return RoundFunction.apply(x, n)


def normalization_on_weights(x, clip_value):
    x = x / clip_value
    # x = torch.where(x.abs() < 1, x, x.sign())
    x = torch.clamp(x, min=-1, max=1)
    return x


def normalization_on_activations(x, clip_value):
    # x = F.relu(x)
    x = x / clip_value
    # x = torch.where(x < 1, x, x.new_ones(x.shape))
    x = torch.clamp(x, min=0, max=1)
    return x


def quantize_activation(
    x,
    bits_activations,
    bits_activations_list,
    clip_value,
    activation_quantization_thresholds,
    is_train=True,
):
    if len(bits_activations_list) == 1 and bits_activations_list[0] == 32:
        return x
    x = normalization_on_activations(x, clip_value)
    ori_input = x

    indicator_list = []
    if is_train:
        output = quantization(ori_input, bits_activations_list[0])
        cum_prod = 1
        for i in range(len(bits_activations_list) - 1):
            residual_error = ori_input - output
            # l1-norm
            residual_error_norm = residual_error.abs().sum() / (output.nelement())
            # l2-norm
            # residual_error_norm = (residual_error ** 2).sum() / (output.nelement())
            indicator = (
                (residual_error_norm > activation_quantization_thresholds[i]).float()
                - torch.sigmoid(
                    residual_error_norm - activation_quantization_thresholds[i]
                ).detach()
                + torch.sigmoid(
                    residual_error_norm - activation_quantization_thresholds[i]
                )
            )
            indicator_list.append(indicator)
            cum_prod = cum_prod * indicator
            inner_output = quantization(residual_error, bits_activations_list[i + 1])
            output = output + cum_prod * inner_output
    else:
        output = quantization(ori_input, bits_activations)

    output = output * clip_value
    return output, indicator_list


def reduce_quantize_activation(
    x,
    bits_activations,
    bits_activations_list,
    clip_value,
    activation_quantization_thresholds,
    is_train=True,
):
    if len(bits_activations_list) == 1 and bits_activations_list[0] == 32:
        return x
    x = normalization_on_activations(x, clip_value)
    ori_input = x

    indicator_list = []
    if is_train:
        output = quantization(ori_input, bits_activations_list[0])
        cum_prod = 1
        for i in range(len(bits_activations_list) - 1):
            residual_error = ori_input - output
            # l1-norm
            residual_error_norm = residual_error.abs().sum() / (output.nelement())
            residual_error_norm = AllReduce.apply(residual_error_norm) * (1.0 / dist.get_world_size())
            # l2-norm
            # residual_error_norm = (residual_error ** 2).sum() / (output.nelement())
            indicator = (
                (residual_error_norm > activation_quantization_thresholds[i]).float()
                - torch.sigmoid(
                    residual_error_norm - activation_quantization_thresholds[i]
                ).detach()
                + torch.sigmoid(
                    residual_error_norm - activation_quantization_thresholds[i]
                )
            )
            indicator_list.append(indicator)
            cum_prod = cum_prod * indicator
            inner_output = quantization(residual_error, bits_activations_list[i + 1])
            output = output + cum_prod * inner_output
    else:
        output = quantization(ori_input, bits_activations)

    output = output * clip_value
    return output, indicator_list


def quantize_weight(
    x,
    bits_weights,
    bits_weights_list,
    clip_value,
    weight_quantization_thresholds,
    is_train=True,
):
    if len(bits_weights_list) == 1 and bits_weights_list[0] == 32:
        return x
    x = normalization_on_weights(x, clip_value)
    x = (x + 1.0) / 2.0
    ori_x = x

    indicator_list = []
    if is_train:
        output = quantization(ori_x, bits_weights_list[0])
        cum_prod = 1
        for i in range(len(bits_weights_list) - 1):
            residual_error = ori_x - output
            # l1-norm
            residual_error_norm = residual_error.abs().sum() / (output.nelement())
            # l2-norm
            # residual_error_norm = (residual_error ** 2).sum() / (output.nelement())
            indicator = (
                (residual_error_norm > weight_quantization_thresholds[i]).float()
                - torch.sigmoid(
                    residual_error_norm - weight_quantization_thresholds[i]
                ).detach()
                + torch.sigmoid(residual_error_norm - weight_quantization_thresholds[i])
            )
            indicator_list.append(indicator)
            cum_prod = cum_prod * indicator
            inner_output = quantization(residual_error, bits_weights_list[i + 1])
            output = output + cum_prod * inner_output
    else:
        output = quantization(ori_x, bits_weights)

    output = output * 2.0 - 1.0
    output = output * clip_value
    return output, indicator_list


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x * n) / n

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class SuperQConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        bits_weights_list=[2, 4, 8],
        bits_activations_list=[2, 4, 8],
    ):
        super(SuperQConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        # self.eps = 1e-5
        self.weight_clip_value = nn.Parameter(torch.Tensor([1.0]))
        self.activation_clip_value = nn.Parameter(torch.Tensor([1.0]))
        self.register_buffer("bits_weights", torch.FloatTensor([2.0]))
        self.register_buffer("bits_activations", torch.FloatTensor([2.0]))
        self.bits_weights_list = bits_weights_list
        self.bits_activations_list = bits_activations_list
        self.weight_quantization_thresholds = nn.Parameter(
            torch.zeros(len(bits_weights_list) - 1)
        )
        self.activation_quantization_thresholds = nn.Parameter(
            torch.zeros(len(bits_activations_list) - 1)
        )
        self.bops = []
        self.delta_bops = []
        for i in range(len(self.bits_weights_list)):
            weight_bops = []
            weight_delta_bops = []
            for j in range(len(self.bits_activations_list)):
                weight_bops.append(0)
                weight_delta_bops.append(0)
            self.bops.append(weight_bops)
            self.delta_bops.append(weight_delta_bops)
        self.input_indicator_list = []
        self.weight_indicator_list = []

    def forward(self, *args):
        if len(args) == 1:
            input = args[0]
            if is_dist_avail_and_initialized():
                quantized_input, self.input_indicator_list = reduce_quantize_activation(
                    input,
                    self.bits_activations,
                    self.bits_activations_list,
                    self.activation_clip_value.abs(),
                    self.activation_quantization_thresholds,
                    self.training,
                )
            else:
                quantized_input, self.input_indicator_list = quantize_activation(
                    input,
                    self.bits_activations,
                    self.bits_activations_list,
                    self.activation_clip_value.abs(),
                    self.activation_quantization_thresholds,
                    self.training,
                )

            weight_mean = self.weight.data.mean()
            weight_std = self.weight.data.std()
            normalized_weight = self.weight.add(-weight_mean).div(weight_std)
            self.quantized_weight, self.weight_indicator_list = quantize_weight(
                normalized_weight,
                self.bits_weights,
                self.bits_weights_list,
                self.weight_clip_value.abs(),
                self.weight_quantization_thresholds,
                self.training,
            )
            output = F.conv2d(
                quantized_input,
                self.quantized_weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            self.h, self.w = output.shape[2], output.shape[3]
        elif len(args) == 2:
            input, weight = args[0], args[1]
            if is_dist_avail_and_initialized():
                quantized_input, self.input_indicator_list = reduce_quantize_activation(
                    input,
                    self.bits_activations,
                    self.bits_activations_list,
                    self.activation_clip_value.abs(),
                    self.activation_quantization_thresholds,
                    self.training,
                )
            else:
                quantized_input, self.input_indicator_list = quantize_activation(
                    input,
                    self.bits_activations,
                    self.bits_activations_list,
                    self.activation_clip_value.abs(),
                    self.activation_quantization_thresholds,
                    self.training,
                )

            weight_mean = weight.data.mean()
            weight_std = weight.data.std()
            normalized_weight = weight.add(-weight_mean).div(weight_std)
            self.quantized_weight, self.weight_indicator_list = quantize_weight(
                normalized_weight,
                self.bits_weights,
                self.bits_weights_list,
                self.weight_clip_value.abs(),
                self.weight_quantization_thresholds,
                self.training,
            )
            output = F.conv2d(
                quantized_input,
                self.quantized_weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            self.h, self.w = output.shape[2], output.shape[3]
        elif len(args) == 4:
            input, weight, weight_mean, weight_std = args[0], args[1], args[2], args[3]
            if is_dist_avail_and_initialized():
                quantized_input, self.input_indicator_list = reduce_quantize_activation(
                    input,
                    self.bits_activations,
                    self.bits_activations_list,
                    self.activation_clip_value.abs(),
                    self.activation_quantization_thresholds,
                    self.training,
                )
            else:
                quantized_input, self.input_indicator_list = quantize_activation(
                    input,
                    self.bits_activations,
                    self.bits_activations_list,
                    self.activation_clip_value.abs(),
                    self.activation_quantization_thresholds,
                    self.training,
                )

            # weight_mean = weight.data.mean()
            # weight_std = weight.data.std()
            normalized_weight = weight.add(-weight_mean).div(weight_std)
            self.quantized_weight, self.weight_indicator_list = quantize_weight(
                normalized_weight,
                self.bits_weights,
                self.bits_weights_list,
                self.weight_clip_value.abs(),
                self.weight_quantization_thresholds,
                self.training,
            )
            output = F.conv2d(
                quantized_input,
                self.quantized_weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            self.h, self.w = output.shape[2], output.shape[3]
        return output

    def pre_compute_bops(self):
        kernel_size = self.kernel_size[0]
        out_channels = self.out_channels
        in_channels = self.in_channels
        for i, bits_weights in enumerate(self.bits_weights_list):
            for j, bits_activations in enumerate(self.bits_activations_list):
                weight_bops = compute_bops(
                    kernel_size,
                    in_channels,
                    out_channels,
                    self.h,
                    self.w,
                    bits_weights,
                    bits_activations,
                )
                self.bops[i][j] = weight_bops
    
        for i, bits_weights in enumerate(self.bits_weights_list):
            for j, bits_activations in enumerate(self.bits_activations_list):
                if j != 0:
                    self.delta_bops[i][j] = self.bops[i][j] - self.bops[i][j - 1]
                else:
                    self.delta_bops[i][j] = self.bops[i][j]

    def given_channel_num_pre_compute_bops(self, channel_num, is_out_channel=True):
        kernel_size = self.kernel_size[0]
        if is_out_channel:
            in_channels = self.in_channels
            out_channels = channel_num
        else:
            in_channels = channel_num
            out_channels = self.out_channels
        for i, bits_weights in enumerate(self.bits_weights_list):
            for j, bits_activations in enumerate(self.bits_activations_list):
                weight_bops = compute_bops(
                    kernel_size,
                    in_channels,
                    out_channels,
                    self.h,
                    self.w,
                    bits_weights,
                    bits_activations,
                )
                self.bops[i][j] = weight_bops
    
        for i, bits_weights in enumerate(self.bits_weights_list):
            for j, bits_activations in enumerate(self.bits_activations_list):
                if j != 0:
                    self.delta_bops[i][j] = self.bops[i][j] - self.bops[i][j - 1]
                else:
                    self.delta_bops[i][j] = self.bops[i][j]

    def compute_bops(self):
        self.pre_compute_bops()
        input_indicator_ = torch.stack(self.input_indicator_list).squeeze()
        input_indicator = torch.cat(
            [input_indicator_.new_tensor([1.0]), input_indicator_]
        )
        cum_input_indicator = torch.cumprod(input_indicator, dim=0)

        weight_indicator_ = torch.stack(self.weight_indicator_list).squeeze()
        weight_indicator = torch.cat(
            [weight_indicator_.new_tensor([1.0]), weight_indicator_]
        )
        cum_weight_indicator = torch.cumprod(weight_indicator, dim=0)

        pre_computed_delta_bops = input_indicator_.new_tensor(self.delta_bops)
        activation_bops = (pre_computed_delta_bops * cum_input_indicator.reshape(1, 3)).sum(dim=1)
        weight_delta_bops = activation_bops[1:] - activation_bops[:-1]
        weight_delta_bops = torch.cat([activation_bops[0].reshape(1), weight_delta_bops])
        bops = (weight_delta_bops * cum_weight_indicator).sum()
        return bops

    def give_channel_num_compute_bops(self, channel_num, is_out_channel=True):
        self.given_channel_num_pre_compute_bops(channel_num, is_out_channel)
        input_indicator_ = torch.stack(self.input_indicator_list).squeeze()
        input_indicator = torch.cat(
            [input_indicator_.new_tensor([1.0]), input_indicator_]
        )
        cum_input_indicator = torch.cumprod(input_indicator, dim=0)

        weight_indicator_ = torch.stack(self.weight_indicator_list).squeeze()
        weight_indicator = torch.cat(
            [weight_indicator_.new_tensor([1.0]), weight_indicator_]
        )
        cum_weight_indicator = torch.cumprod(weight_indicator, dim=0)

        pre_computed_delta_bops = input_indicator_.new_tensor(self.delta_bops)
        activation_bops = (pre_computed_delta_bops * cum_input_indicator.reshape(1, 3)).sum(dim=1)
        weight_delta_bops = activation_bops[1:] - activation_bops[:-1]
        weight_delta_bops = torch.cat([activation_bops[0].reshape(1), weight_delta_bops])
        bops = (weight_delta_bops * cum_weight_indicator).sum()
        return bops

    def compute_current_bops(self, channel_num, is_out_channel=True):
        if self.input_indicator_list[0] == 0:
            self.bits_activations.data.fill_(2)
        elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 0:
            self.bits_activations.data.fill_(4)
        elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 1:
            self.bits_activations.data.fill_(8)

        if self.weight_indicator_list[0] == 0.0:
            self.bits_weights.data.fill_(2)
        elif self.weight_indicator_list[0] == 1.0 and self.weight_indicator_list[1] == 0.0:
            self.bits_weights.data.fill_(4)
        elif self.weight_indicator_list[0] == 1.0 and self.weight_indicator_list[1] == 1.0:
            self.bits_weights.data.fill_(8)

        if is_out_channel:
            out_channels = channel_num
            in_channels = self.in_channels
        else:
            out_channels = self.out_channels
            in_channels = channel_num

        kernel_size = self.kernel_size[0]
        bops = compute_bops(
            kernel_size, 
            in_channels,
            out_channels,
            self.h,
            self.w,
            self.bits_weights.data.item(),
            self.bits_activations.data.item())
        return bops

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights_list={}".format(self.bits_weights_list)
        s += ", bits_activations_list={}".format(self.bits_activations_list)
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("dorefa_clip_rcf_wn_super_joint_quan_conv")
        return s