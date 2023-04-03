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
    x = torch.clamp(x, min=-1, max=1)
    return x


def normalization_on_activations(x, clip_value, bias):
    x = (x - bias) / clip_value
    x = torch.clamp(x, min=0, max=1)
    return x


def quantize_activation(
    x,
    bits_activations,
    bits_activations_list,
    clip_value,
    bias,
    activation_quantization_thresholds,
    is_train=True,
):
    if len(bits_activations_list) == 1 and bits_activations_list[0] == 32:
        return x
    x = normalization_on_activations(x, clip_value, bias)
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

    output = output * clip_value + bias
    return output, indicator_list


def reduce_quantize_activation(
    x,
    bits_activations,
    bits_activations_list,
    clip_value,
    bias,
    activation_quantization_thresholds,
    is_train=True,
):
    if len(bits_activations_list) == 1 and bits_activations_list[0] == 32:
        return x
    x = normalization_on_activations(x, clip_value, bias)
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

    output = output * clip_value + bias
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


def normalize_and_quantize_weight(
    x,
    bits_weights,
    clip_value):
    x = normalization_on_weights(x, clip_value)
    x = (x + 1.0) / 2.0
    output = quantization(x, bits_weights)
    return output


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x * n) / n

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class SuperAsyQConv2d(nn.Conv2d):
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
        super(SuperAsyQConv2d, self).__init__(
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
        self.activation_bias = nn.Parameter(torch.FloatTensor([0.0]))
        self.register_buffer("bits_weights", torch.FloatTensor([8.0]))
        self.register_buffer("bits_activations", torch.FloatTensor([8.0]))
        self.bits_weights_list = bits_weights_list
        self.bits_activations_list = bits_activations_list
        self.weight_quantization_thresholds = nn.Parameter(
            torch.zeros(len(bits_weights_list) - 1)
        )
        self.activation_quantization_thresholds = nn.Parameter(
            torch.zeros(len(bits_activations_list) - 1)
        )
        self.weight_bops = []
        for i in range(len(self.bits_weights_list)):
            self.weight_bops.append(0)
        self.activation_bops = []
        for i in range(len(self.bits_activations_list)):
            self.activation_bops.append(0)
        self.input_indicator_list = []
        self.weight_indicator_list = []
        self.num_bits = len(self.bits_weights_list)

    def forward(self, *args):
        if len(args) == 1:
            input = args[0]
            if is_dist_avail_and_initialized():
                quantized_input, self.input_indicator_list = reduce_quantize_activation(
                    input,
                    self.bits_activations,
                    self.bits_activations_list,
                    self.activation_clip_value.abs(),
                    self.activation_bias,
                    self.activation_quantization_thresholds,
                    self.training,
                )
            else:
                quantized_input, self.input_indicator_list = quantize_activation(
                    input,
                    self.bits_activations,
                    self.bits_activations_list,
                    self.activation_clip_value.abs(),
                    self.activation_bias,
                    self.activation_quantization_thresholds,
                    self.training,
                )

            self.quantized_weight, self.weight_indicator_list = quantize_weight(
                self.weight,
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
                    self.activation_bias,
                    self.activation_quantization_thresholds,
                    self.training,
                )
            else:
                quantized_input, self.input_indicator_list = quantize_activation(
                    input,
                    self.bits_activations,
                    self.bits_activations_list,
                    self.activation_clip_value.abs(),
                    self.activation_bias,
                    self.activation_quantization_thresholds,
                    self.training,
                )

            self.quantized_weight, self.weight_indicator_list = quantize_weight(
                self.weight,
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

    def pre_compute_bops(
        self,
        given_bit_weights,
        given_bit_activation,
        channel_num,
        is_compute_weights_bops=True,
        is_out_channel=True,
    ):
        kernel_size = self.kernel_size[0]
        if is_out_channel:
            out_channels = channel_num
            in_channels = self.in_channels
        else:
            out_channels = self.out_channels
            in_channels = channel_num
        if is_compute_weights_bops:
            for i, bits_weights in enumerate(self.bits_weights_list):
                self.weight_bops[i] = compute_bops(
                    kernel_size,
                    in_channels,
                    out_channels // self.groups,
                    self.h,
                    self.w,
                    bits_weights,
                    given_bit_activation,
                )
            for i in range(len(self.bits_weights_list) - 1, -1, -1):
                if i != 0:
                    self.weight_bops[i] = self.weight_bops[i] - self.weight_bops[i - 1]
        else:
            for i, bits_activations in enumerate(self.bits_activations_list):
                self.activation_bops[i] = compute_bops(
                    kernel_size,
                    in_channels,
                    out_channels // self.groups,
                    self.h,
                    self.w,
                    given_bit_weights,
                    bits_activations,
                )
            for i in range(len(self.bits_weights_list) - 1, -1, -1):
                if i != 0:
                    self.activation_bops[i] = (
                        self.activation_bops[i] - self.activation_bops[i - 1]
                    )

    def pre_compute_channel_bops(
        self,
        given_bit_weights,
        given_bit_activation,
        in_channels,
        out_channels,
        is_compute_weights_bops=True,
    ):
        kernel_size = self.kernel_size[0]
        if is_compute_weights_bops:
            for i, bits_weights in enumerate(self.bits_weights_list):
                self.weight_bops[i] = compute_bops(
                    kernel_size,
                    in_channels,
                    out_channels // self.groups,
                    self.h,
                    self.w,
                    bits_weights,
                    given_bit_activation,
                )
            for i in range(len(self.bits_weights_list) - 1, -1, -1):
                if i != 0:
                    self.weight_bops[i] = self.weight_bops[i] - self.weight_bops[i - 1]
        else:
            for i, bits_activations in enumerate(self.bits_activations_list):
                self.activation_bops[i] = compute_bops(
                    kernel_size,
                    in_channels,
                    out_channels // self.groups,
                    self.h,
                    self.w,
                    given_bit_weights,
                    bits_activations,
                )
            for i in range(len(self.bits_weights_list) - 1, -1, -1):
                if i != 0:
                    self.activation_bops[i] = (
                        self.activation_bops[i] - self.activation_bops[i - 1]
                    )

    def fix_activation_pre_compute_weight_bops(self, bit_activations):
        kernel_size = self.kernel_size[0]
        for i, bits_weights in enumerate(self.bits_weights_list):
            self.weight_bops[i] = compute_bops(
                kernel_size,
                self.in_channels,
                self.out_channels // self.groups,
                self.h,
                self.w,
                bits_weights,
                bit_activations,
            )
        for i in range(len(self.bits_weights_list) - 1, -1, -1):
            if i != 0:
                self.weight_bops[i] = self.weight_bops[i] - self.weight_bops[i - 1]

    def fix_weight_pre_compute_activation_bops(self, bit_weights):
        kernel_size = self.kernel_size[0]
        for i, bits_activations in enumerate(self.bits_activations_list):
            self.activation_bops[i] = compute_bops(
                kernel_size,
                self.in_channels,
                self.out_channels // self.groups,
                self.h,
                self.w,
                bit_weights,
                bits_activations,
            )
        for i in range(len(self.bits_weights_list) - 1, -1, -1):
            if i != 0:
                self.activation_bops[i] = (
                    self.activation_bops[i] - self.activation_bops[i - 1]
                )

    def fix_weight_compute_activation_bops(self):
        if self.num_bits == 3:
            if self.weight_indicator_list[0] == 0.0:
                self.bits_weights.data.fill_(2)
            elif (
                self.weight_indicator_list[0] == 1.0
                and self.weight_indicator_list[1] == 0.0
            ):
                self.bits_weights.data.fill_(4)
            elif (
                self.weight_indicator_list[0] == 1.0
                and self.weight_indicator_list[1] == 1.0
            ):
                self.bits_weights.data.fill_(8)
        else:
            if self.weight_indicator_list[0] == 0.0:
                self.bits_weights.data.fill_(4)
            elif self.weight_indicator_list[0] == 1.0:
                self.bits_weights.data.fill_(8)
        self.fix_weight_pre_compute_activation_bops(self.bits_weights)

        input_indicator_ = torch.stack(self.input_indicator_list)
        if self.num_bits == 3:
            input_indicator_ = input_indicator_.squeeze()
        input_indicator = torch.cat(
            [input_indicator_.new_tensor([1.0]), input_indicator_]
        )

        activation_bops = input_indicator_.new_tensor(self.activation_bops)
        bops = (torch.cumprod(input_indicator, dim=0) * activation_bops).sum()
        return bops

    def compress_fix_weight_compute_activation_bops(
        self, channel_num, is_out_channel=True
    ):
        if self.num_bits == 3:
            if self.weight_indicator_list[0] == 0.0:
                self.bits_weights.data.fill_(2)
            elif (
                self.weight_indicator_list[0] == 1.0
                and self.weight_indicator_list[1] == 0.0
            ):
                self.bits_weights.data.fill_(4)
            elif (
                self.weight_indicator_list[0] == 1.0
                and self.weight_indicator_list[1] == 1.0
            ):
                self.bits_weights.data.fill_(8)
        else:
            if self.weight_indicator_list[0] == 0.0:
                self.bits_weights.data.fill_(4)
            elif self.weight_indicator_list[0] == 1.0:
                self.bits_weights.data.fill_(8)
        self.pre_compute_bops(
            self.bits_weights,
            8,
            channel_num,
            is_compute_weights_bops=False,
            is_out_channel=is_out_channel,
        )

        input_indicator_ = torch.stack(self.input_indicator_list)
        if self.num_bits == 3:
            input_indicator_ = input_indicator_.squeeze()
        input_indicator = torch.cat(
            [input_indicator_.new_tensor([1.0]), input_indicator_]
        )

        activation_bops = input_indicator_.new_tensor(self.activation_bops)
        bops = (torch.cumprod(input_indicator, dim=0) * activation_bops).sum()
        return bops

    def compress_channel_fix_weight_compute_activation_bops(
        self, in_channels, out_channels
    ):
        if self.num_bits == 3:
            if self.weight_indicator_list[0] == 0.0:
                self.bits_weights.data.fill_(2)
            elif (
                self.weight_indicator_list[0] == 1.0
                and self.weight_indicator_list[1] == 0.0
            ):
                self.bits_weights.data.fill_(4)
            elif (
                self.weight_indicator_list[0] == 1.0
                and self.weight_indicator_list[1] == 1.0
            ):
                self.bits_weights.data.fill_(8)
        else:
            if self.weight_indicator_list[0] == 0.0:
                self.bits_weights.data.fill_(4)
            elif self.weight_indicator_list[0] == 1.0:
                self.bits_weights.data.fill_(8)
        self.pre_compute_channel_bops(
            self.bits_weights,
            8,
            in_channels,
            out_channels,
            is_compute_weights_bops=False,
        )

        input_indicator_ = torch.stack(self.input_indicator_list)
        if self.num_bits == 3:
            input_indicator_ = input_indicator_.squeeze()
        input_indicator = torch.cat(
            [input_indicator_.new_tensor([1.0]), input_indicator_]
        )

        activation_bops = input_indicator_.new_tensor(self.activation_bops)
        bops = (torch.cumprod(input_indicator, dim=0) * activation_bops).sum()
        return bops

    def fix_activation_compute_weight_bops(self):
        if self.num_bits == 3:
            if self.input_indicator_list[0] == 0:
                self.bits_activations.data.fill_(2)
            elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 0:
                self.bits_activations.data.fill_(4)
            elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 1:
                self.bits_activations.data.fill_(8)
        else:
            if self.input_indicator_list[0] == 0:
                self.bits_activations.data.fill_(4)
            elif self.input_indicator_list[0] == 1:
                self.bits_activations.data.fill_(8)
        self.fix_activation_pre_compute_weight_bops(self.bits_activations)

        weight_indicator_ = torch.stack(self.weight_indicator_list)
        if self.num_bits == 3:
            weight_indicator_ = weight_indicator_.squeeze()
        weight_indicator = torch.cat(
            [weight_indicator_.new_tensor([1.0]), weight_indicator_]
        )

        weight_bops = weight_indicator_.new_tensor(self.weight_bops)
        bops = (torch.cumprod(weight_indicator, dim=0) * weight_bops).sum()
        return bops

    def compress_fix_activation_compute_weight_bops(
        self, channel_num, is_out_channel=True
    ):
        if self.num_bits == 3:
            if self.input_indicator_list[0] == 0:
                self.bits_activations.data.fill_(2)
            elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 0:
                self.bits_activations.data.fill_(4)
            elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 1:
                self.bits_activations.data.fill_(8)
        else:
            if self.input_indicator_list[0] == 0:
                self.bits_activations.data.fill_(4)
            elif self.input_indicator_list[0] == 1:
                self.bits_activations.data.fill_(8)
        self.pre_compute_bops(
            8,
            self.bits_activations,
            channel_num,
            is_compute_weights_bops=True,
            is_out_channel=is_out_channel,
        )

        weight_indicator_ = torch.stack(self.weight_indicator_list)
        if self.num_bits == 3:
            weight_indicator_ = weight_indicator_.squeeze()
        weight_indicator = torch.cat(
            [weight_indicator_.new_tensor([1.0]), weight_indicator_]
        )

        weight_bops = weight_indicator_.new_tensor(self.weight_bops)
        bops = (torch.cumprod(weight_indicator, dim=0) * weight_bops).sum()
        return bops

    def compress_channel_fix_activation_compute_weight_bops(
        self, in_channels, out_channels
    ):
        if self.num_bits == 3:
            if self.input_indicator_list[0] == 0:
                self.bits_activations.data.fill_(2)
            elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 0:
                self.bits_activations.data.fill_(4)
            elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 1:
                self.bits_activations.data.fill_(8)
        else:
            if self.input_indicator_list[0] == 0:
                self.bits_activations.data.fill_(4)
            elif self.input_indicator_list[0] == 1:
                self.bits_activations.data.fill_(8)
        self.pre_compute_channel_bops(
            8,
            self.bits_activations,
            in_channels,
            out_channels,
            is_compute_weights_bops=True,
        )

        weight_indicator_ = torch.stack(self.weight_indicator_list)
        if self.num_bits == 3:
            weight_indicator_ = weight_indicator_.squeeze()
        weight_indicator = torch.cat(
            [weight_indicator_.new_tensor([1.0]), weight_indicator_]
        )

        weight_bops = weight_indicator_.new_tensor(self.weight_bops)
        bops = (torch.cumprod(weight_indicator, dim=0) * weight_bops).sum()
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
            out_channels // self.groups,
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
        s += ", method={}".format("dorefa_clip_rcf_super_quan_asy_conv")
        return s


class SuperAsyQLinear(nn.Linear):
    """
    custom convolutional layers for quantization
    """

    def __init__(
        self, in_features, out_features, bias=True, 
        bits_weights_list=[2, 4, 8],
        bits_activations_list=[2, 4, 8],
    ):
        super(SuperAsyQLinear, self).__init__(in_features, out_features, bias=bias)
        # self.eps = 1e-5
        self.init_state = False
        self.weight_clip_value = nn.Parameter(torch.Tensor([1]))
        self.activation_clip_value = nn.Parameter(torch.Tensor([1]))
        self.activation_bias = nn.Parameter(torch.Tensor([0.0]))
        self.register_buffer("bits_weights", torch.FloatTensor([8.0]))
        self.register_buffer("bits_activations", torch.FloatTensor([8.0]))
        self.bits_weights_list = bits_weights_list
        self.bits_activations_list = bits_activations_list
        self.weight_quantization_thresholds = nn.Parameter(
            torch.zeros(len(bits_weights_list) - 1)
        )
        self.activation_quantization_thresholds = nn.Parameter(
            torch.zeros(len(bits_activations_list) - 1)
        )
        self.weight_bops = []
        for i in range(len(self.bits_weights_list)):
            self.weight_bops.append(0)
        self.activation_bops = []
        for i in range(len(self.bits_activations_list)):
            self.activation_bops.append(0)
        self.input_indicator_list = []
        self.weight_indicator_list = []
        self.num_bits = len(self.bits_weights_list)

    def forward(self, input):
        quantized_input, self.input_indicator_list = quantize_activation(
            input, 
            self.bits_activations, 
            self.bits_activations_list,
            self.activation_clip_value.abs(),
            self.activation_bias,
            self.activation_quantization_thresholds,
            self.training
            )
        self.quantized_weight, self.weight_indicator_list = quantize_weight(
            self.weight, 
            self.bits_weights, 
            self.bits_weights_list,
            self.weight_clip_value.abs(),
            self.weight_quantization_thresholds,
            self.training
            )
        output = F.linear(quantized_input, self.quantized_weight, self.bias)
        self.output_shape = output.shape
        return output

    def pre_compute_bops(
        self,
        given_bit_weights,
        given_bit_activation,
        channel_num,
        is_compute_weights_bops=True,
        is_out_channel=True,
    ):
        kernel_size = 1
        if is_out_channel:
            out_channels = channel_num
            in_channels = self.in_features
        else:
            out_channels = self.out_features
            in_channels = channel_num
        if is_compute_weights_bops:
            for i, bits_weights in enumerate(self.bits_weights_list):
                self.weight_bops[i] = compute_bops(
                    kernel_size,
                    in_channels,
                    out_channels,
                    1,
                    1,
                    bits_weights,
                    given_bit_activation,
                )
            for i in range(len(self.bits_weights_list) - 1, -1, -1):
                if i != 0:
                    self.weight_bops[i] = self.weight_bops[i] - self.weight_bops[i - 1]
        else:
            for i, bits_activations in enumerate(self.bits_activations_list):
                self.activation_bops[i] = compute_bops(
                    kernel_size,
                    in_channels,
                    out_channels,
                    1,
                    1,
                    given_bit_weights,
                    bits_activations,
                )
            for i in range(len(self.bits_weights_list) - 1, -1, -1):
                if i != 0:
                    self.activation_bops[i] = (
                        self.activation_bops[i] - self.activation_bops[i - 1]
                    )

    def fix_activation_pre_compute_weight_bops(self, bit_activations):
        for i, bits_weights in enumerate(self.bits_weights_list):
            self.weight_bops[i] = compute_bops(
                1,
                self.in_features,
                self.out_features,
                1,
                1,
                bits_weights,
                bit_activations,
            )
        for i in range(len(self.bits_weights_list) - 1, -1, -1):
            if i != 0:
                self.weight_bops[i] = self.weight_bops[i] - self.weight_bops[i - 1]
    
    def fix_weight_pre_compute_activation_bops(self, bit_weights):
        for i, bits_activations in enumerate(self.bits_activations_list):
            self.activation_bops[i] = compute_bops(
                1,
                self.in_features,
                self.out_features,
                1,
                1,
                bit_weights,
                bits_activations,
            )
        for i in range(len(self.bits_activations_list) - 1, -1, -1):
            if i != 0:
                self.activation_bops[i] = (
                    self.activation_bops[i] - self.activation_bops[i - 1]
                )
    
    def fix_weight_compute_activation_bops(self):
        if self.weight_indicator_list[0] == 0.0:
            self.bits_weights.data.fill_(2)
        elif (
            self.weight_indicator_list[0] == 1.0
            and self.weight_indicator_list[1] == 0.0
        ):
            self.bits_weights.data.fill_(4)
        elif (
            self.weight_indicator_list[0] == 1.0
            and self.weight_indicator_list[1] == 1.0
        ):
            self.bits_weights.data.fill_(8)
        self.fix_weight_pre_compute_activation_bops(self.bits_weights)

        input_indicator_ = torch.stack(self.input_indicator_list).squeeze()
        input_indicator = torch.cat(
            [input_indicator_.new_tensor([1.0]), input_indicator_]
        )

        if isinstance(self.activation_bops[0], float):
            for i in range(len(self.activation_bops)):
                self.activation_bops[i] = input_indicator_.new_tensor(
                    [self.activation_bops[i]]
                )

        activation_bops = torch.stack(self.activation_bops).squeeze()
        bops = (torch.cumprod(input_indicator, dim=0) * activation_bops).sum()
        return bops
    
    def fix_activation_compute_weight_bops(self):
        if self.input_indicator_list[0] == 0:
            self.bits_activations.data.fill_(2)
        elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 0:
            self.bits_activations.data.fill_(4)
        elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 1:
            self.bits_activations.data.fill_(8)
        self.fix_activation_pre_compute_weight_bops(self.bits_activations)

        weight_indicator_ = torch.stack(self.weight_indicator_list).squeeze()
        weight_indicator = torch.cat(
            [weight_indicator_.new_tensor([1.0]), weight_indicator_]
        )

        if isinstance(self.weight_bops[0], float):
            for i in range(len(self.weight_bops)):
                self.weight_bops[i] = weight_indicator_.new_tensor(
                    [self.weight_bops[i]]
                )

        weight_bops = torch.stack(self.weight_bops).squeeze()
        bops = (torch.cumprod(weight_indicator, dim=0) * weight_bops).sum()
        return bops

    def compress_fix_weight_compute_activation_bops(
        self, channel_num, is_out_channel=True
    ):
        if self.weight_indicator_list[0] == 0.0:
            self.bits_weights.data.fill_(2)
        elif (
            self.weight_indicator_list[0] == 1.0
            and self.weight_indicator_list[1] == 0.0
        ):
            self.bits_weights.data.fill_(4)
        elif (
            self.weight_indicator_list[0] == 1.0
            and self.weight_indicator_list[1] == 1.0
        ):
            self.bits_weights.data.fill_(8)
        self.pre_compute_bops(
            self.bits_weights,
            8,
            channel_num,
            is_compute_weights_bops=False,
            is_out_channel=is_out_channel,
        )

        input_indicator_ = torch.stack(self.input_indicator_list)
        input_indicator = torch.cat(
            [input_indicator_.new_tensor([1.0]), input_indicator_]
        )

        activation_bops = input_indicator_.new_tensor(self.activation_bops)
        bops = (torch.cumprod(input_indicator, dim=0) * activation_bops).sum()
        return bops

    def compress_fix_activation_compute_weight_bops(
        self, channel_num, is_out_channel=True
    ):
        if self.input_indicator_list[0] == 0:
            self.bits_activations.data.fill_(2)
        elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 0:
            self.bits_activations.data.fill_(4)
        elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 1:
            self.bits_activations.data.fill_(8)
        self.pre_compute_bops(
            8,
            self.bits_activations,
            channel_num,
            is_compute_weights_bops=True,
            is_out_channel=is_out_channel,
        )

        weight_indicator_ = torch.stack(self.weight_indicator_list)
        weight_indicator = torch.cat(
            [weight_indicator_.new_tensor([1.0]), weight_indicator_]
        )

        weight_bops = weight_indicator_.new_tensor(self.weight_bops)
        bops = (torch.cumprod(weight_indicator, dim=0) * weight_bops).sum()
        return bops
    
    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights_list={}".format(self.bits_weights_list)
        s += ", bits_activations_list={}".format(self.bits_activations_list)
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("dorefa_rcf_asy_linear")
        return s