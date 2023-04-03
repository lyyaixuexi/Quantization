import math

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F


def quantization(x, k):
    n = 2 ** k - 1
    return RoundFunction.apply(x, n)


def normalization_on_weights(x, clip_value):
    x = x / clip_value
    x = torch.where(x.abs() < 1, x, x.sign())
    return x


def normalization_on_activations(x, clip_value):
    x = F.relu(x)
    x = x / clip_value
    x = torch.where(x < 1, x, x.new_ones(x.shape))
    return x


def quantize_activation(
    x, bits_activations, bits_activations_list, clip_value, activation_quantization_thresholds, is_train=True
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
            residual_error_norm = (residual_error ** 2).sum() / (output.nelement())
            indicator = (
                (residual_error_norm > activation_quantization_thresholds[i]).float()
                - torch.sigmoid(residual_error_norm - activation_quantization_thresholds[i]).detach()
                + torch.sigmoid(residual_error_norm - activation_quantization_thresholds[i])
            )
            indicator_list.append(indicator)
            cum_prod = cum_prod * indicator
            inner_output = quantization(residual_error, bits_activations_list[i + 1])
            output = output + cum_prod * inner_output
    else:
        output = quantization(ori_input, bits_activations)
    
    output = output * clip_value
    return output, indicator_list


def quantize_weight(x, bits_weights, bits_weights_list, clip_value, weight_quantization_thresholds, is_train=True):
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
            residual_error_norm = (residual_error ** 2).sum() / (output.nelement())
            indicator = (
                (residual_error_norm > weight_quantization_thresholds[i]).float()
                - torch.sigmoid(residual_error_norm - weight_quantization_thresholds[i]).detach()
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
        self.weight_bops = []
        for i in range(len(self.bits_weights_list)):
            self.weight_bops.append(0)
        self.activation_bops = []
        for i in range(len(self.bits_activations_list)):
            self.activation_bops.append(0)
        self.input_indicator_list = []
        self.weight_indicator_list = []


    def forward(self, input, filter_mask=[]):
        quantized_input, self.input_indicator_list = quantize_activation(
            input,
            self.bits_activations,
            self.bits_activations_list,
            self.activation_clip_value.abs(),
            self.activation_quantization_thresholds,
            self.training
        )

        if len(filter_mask) != 0:
            self.selected_weight = torch.index_select(self.weight, 1, filter_mask.nonzero().reshape(-1))
            weight_mean = self.selected_weight.data.mean()
            weight_std = self.selected_weight.data.std()
        else:
            weight_mean = self.weight.data.mean()
            weight_std = self.weight.data.std()
        normalized_weight = self.weight.add(-weight_mean).div(weight_std)
        quantized_weight, self.weight_indicator_list = quantize_weight(
            normalized_weight,
            self.bits_weights,
            self.bits_weights_list,
            self.weight_clip_value.abs(),
            self.weight_quantization_thresholds,
            self.training
        )
        if len(filter_mask) != 0:
            quantized_weight = quantized_weight * filter_mask.reshape(1, filter_mask.shape[0], 1, 1)
        output = F.conv2d(
            quantized_input,
            quantized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        self.h, self.w = output.shape[2], output.shape[3]
        return output

    def fix_activation_pre_compute_weight_bops(self, bit_activations, in_channels):
        kernel_size = self.kernel_size[0]
        for i, bits_weights in enumerate(self.bits_weights_list):
            nk_square = in_channels * kernel_size * kernel_size
            log_nk_square = torch.log2(nk_square) if isinstance(nk_square, torch.Tensor) else math.log(nk_square, 2)
            bop = (
                self.out_channels
                * nk_square
                * (
                    bits_weights * bit_activations
                    + bit_activations
                    + bits_weights
                    + log_nk_square
                )
                * self.h
                * self.w
            )
            self.weight_bops[i] = bop
        for i in range(len(self.bits_weights_list) - 1, -1, -1):
            if i != 0:
                self.weight_bops[i] = self.weight_bops[i] - self.weight_bops[i - 1]

    def fix_weight_pre_compute_activation_bops(self, bit_weights, in_channels):
        kernel_size = self.kernel_size[0]
        for i, bits_activations in enumerate(self.bits_activations_list):
            nk_square = in_channels * kernel_size * kernel_size
            log_nk_square = torch.log2(nk_square) if isinstance(nk_square, torch.Tensor) else math.log(nk_square, 2)
            bop = (
                self.out_channels
                * nk_square
                * (
                    bit_weights * bits_activations
                    + bits_activations
                    + bit_weights
                    + log_nk_square
                )
                * self.h
                * self.w
            )
            self.activation_bops[i] = bop
        for i in range(len(self.bits_weights_list) - 1, -1, -1):
            if i != 0:
                self.activation_bops[i] = (
                    self.activation_bops[i] - self.activation_bops[i - 1]
                )

    def fix_weight_compute_activation_bops(self, in_channels):
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
        self.fix_weight_pre_compute_activation_bops(self.bits_weights, in_channels)

        input_indicator_ = torch.stack(self.input_indicator_list).squeeze()
        input_indicator = torch.cat([input_indicator_.new_tensor([1.0]), input_indicator_])

        if isinstance(self.activation_bops[0], float):
            for i in range(len(self.activation_bops)):
                self.activation_bops[i] = input_indicator_.new_tensor([self.activation_bops[i]])

        activation_bops = torch.stack(self.activation_bops).squeeze()
        bops = (torch.cumprod(input_indicator, dim=0) * activation_bops).sum()
        return bops

    def fix_activation_compute_weight_bops(self, in_channels):
        if self.input_indicator_list[0] == 0:
            self.bits_activations.data.fill_(2)
        elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 0:
            self.bits_activations.data.fill_(4)
        elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 1:
            self.bits_activations.data.fill_(8)
        self.fix_activation_pre_compute_weight_bops(self.bits_activations, in_channels)

        weight_indicator_ = torch.stack(self.weight_indicator_list).squeeze()
        weight_indicator = torch.cat([weight_indicator_.new_tensor([1.0]), weight_indicator_])

        if isinstance(self.weight_bops[0], float):
            for i in range(len(self.weight_bops)):
                self.weight_bops[i] = weight_indicator_.new_tensor([self.weight_bops[i]])

        weight_bops = torch.stack(self.weight_bops).squeeze()
        bops = (torch.cumprod(weight_indicator, dim=0) * weight_bops).sum()
        return bops

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights_list={}".format(self.bits_weights_list)
        s += ", bits_activations_list={}".format(self.bits_activations_list)
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("dorefa_clip_rcf_wn_super_quan_conv")
        return s


class SuperCompressQConv2d(nn.Conv2d):
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
        super(SuperCompressQConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.eps = 1e-5
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

        # self.pruning_thresholds = nn.Parameter(torch.zeros(1))
        self.pruning_theta = nn.Parameter(torch.ones(self.weight.shape[0]) * 0.1)
        self.register_buffer("mask", torch.ones(self.weight.shape[0]))

        self.weight_bops = []
        for i in range(len(self.bits_weights_list)):
            self.weight_bops.append(0)
        self.activation_bops = []
        for i in range(len(self.bits_activations_list)):
            self.activation_bops.append(0)
        self.input_indicator_list = []
        self.weight_indicator_list = []


    def forward(self, input):
        quantized_input, self.input_indicator_list = quantize_activation(
            input,
            self.bits_activations,
            self.bits_activations_list,
            self.activation_clip_value,
            self.activation_quantization_thresholds,
            self.training
        )

        self.compute_mask()

        self.selected_weight = torch.index_select(self.weight, 0, self.mask.nonzero().reshape(-1))
        weight_mean = self.selected_weight.data.mean()
        weight_std = self.selected_weight.data.std()
        normalized_weight = self.weight.add(-weight_mean).div(weight_std)
        quantized_weight, self.weight_indicator_list = quantize_weight(
            normalized_weight,
            self.bits_weights,
            self.bits_weights_list,
            self.weight_clip_value,
            self.weight_quantization_thresholds,
            self.training
        )
        quantized_weight = quantized_weight * self.mask.reshape(self.mask.shape[0], 1, 1, 1)
        output = F.conv2d(
            quantized_input,
            quantized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        self.h, self.w = output.shape[2], output.shape[3]
        return output, self.mask

    def compute_mask(self):
        self.mask = (
            (self.pruning_theta > 0).float()
            - torch.sigmoid(self.pruning_theta).detach()
            + torch.sigmoid(self.pruning_theta)
        )
        # print(self.pruning_theta)
        # print(self.mask)

    def fix_activation_pre_compute_weight_bops(self, bit_activations):
        kernel_size = self.kernel_size[0]
        for i, bits_weights in enumerate(self.bits_weights_list):
            nk_square = self.in_channels * kernel_size * kernel_size
            bop = (
                self.out_channels
                * nk_square
                * (
                    bits_weights * bit_activations
                    + bit_activations
                    + bits_weights
                    + math.log(nk_square, 2)
                )
                * self.h
                * self.w
            )
            self.weight_bops[i] = bop
        for i in range(len(self.bits_weights_list) - 1, -1, -1):
            if i != 0:
                self.weight_bops[i] = self.weight_bops[i] - self.weight_bops[i - 1]

    def fix_weight_pre_compute_activation_bops(self, bit_weights):
        kernel_size = self.kernel_size[0]
        for i, bits_activations in enumerate(self.bits_activations_list):
            nk_square = self.in_channels * kernel_size * kernel_size
            bop = (
                self.out_channels
                * nk_square
                * (
                    bit_weights * bits_activations
                    + bits_activations
                    + bit_weights
                    + math.log(nk_square, 2)
                )
                * self.h
                * self.w
            )
            self.activation_bops[i] = bop
        for i in range(len(self.bits_weights_list) - 1, -1, -1):
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
        bops = self.activation_bops[0]
        product = 1
        for i in range(len(self.input_indicator_list)):
            product *= self.input_indicator_list[i]
            bops = bops + product * self.activation_bops[i + 1]
        filter_num = self.mask.sum()
        filter_ratio = filter_num / self.out_channels
        return bops * filter_ratio

    def fix_activation_compute_weight_bops(self):
        if self.input_indicator_list[0] == 0:
            self.bits_activations.data.fill_(2)
        elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 0:
            self.bits_activations.data.fill_(4)
        elif self.input_indicator_list[0] == 1 and self.input_indicator_list[1] == 1:
            self.bits_activations.data.fill_(8)
        self.fix_activation_pre_compute_weight_bops(self.bits_activations)
        bops = self.weight_bops[0]
        product = 1
        for i in range(len(self.weight_indicator_list)):
            product *= self.weight_indicator_list[i]
            bops = bops + product * self.weight_bops[i + 1]
        filter_num = self.mask.sum()
        filter_ratio = filter_num / self.out_channels
        return bops * filter_ratio

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights_list={}".format(self.bits_weights_list)
        s += ", bits_activations_list={}".format(self.bits_activations_list)
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("dorefa_clip_rcf_wn_super_compressed_conv")
        return s
