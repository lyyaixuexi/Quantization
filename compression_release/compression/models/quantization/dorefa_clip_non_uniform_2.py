import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F


# def quantization(x, k):
#     n = 2 ** k - 1
#     return RoundFunction_2.apply(x, n)


def normalization_on_weights(x, clip_value):
    x = x / clip_value
    x = torch.where(x.abs() < 1, x, x.sign())
    return x


def normalization_on_weights_negative(x, clip_value):
    x = x / clip_value
    x = torch.clamp(x, min=-1.0, max=0.0)
    return x


def normalization_on_weights_positive(x, clip_value):
    x = x / clip_value
    x = torch.clamp(x, min=0.0, max=1.0)
    return x


def normalization_on_activations(x, clip_value):
    x = x * clip_value
    x = torch.clamp(x, min=0, max=1)
    return x


def quantization(x, k, bias):
    n = 2.0 ** k - 1.0
    sum_y = 0
    bias_sum = 0
    for i in range(int(n)):
        x_i = x - bias_sum
        y = normalization_on_activations(x_i, bias[i])
        y = RoundFunction.apply(y)
        sum_y += y
        bias_sum += 1.0/bias[i]
    sum_y = sum_y / n
    return sum_y


def quantize_activation(x, k, clip_value, activation_bias):
    if k == 32:
        return x

    x = x / clip_value
    x = quantization(x, k, activation_bias)
    x = x * clip_value
    return x


def quantize_weight(x, k, clip_value, weight_bias):
    if k == 32:
        return x
    x = normalization_on_weights(x, clip_value)
    x = (x + 1.0) / 2.0

    x = quantization(x, k, weight_bias)

    # x = quantization(x, k)
    x = x * 2.0 - 1.0
    x = x * clip_value
    return x


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class RoundFunction_2(Function):
    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x * n) / n

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class QConv2d(nn.Conv2d):
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
        bits_weights=32,
        bits_activations=32,
    ):
        super(QConv2d, self).__init__(
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
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations
        self.weight_n = 2 ** bits_weights - 1
        self.activation_n = 2 ** bits_activations - 1

        self.weight_level = []
        self.activation_level = []
        if bits_weights != 32:
            for i in range(self.weight_n):
                self.weight_level.append(self.weight_n)

        if bits_activations != 32:
            for i in range(self.activation_n):
                self.activation_level.append(1.0)

        # self.eps = 1e-5
        self.weight_bias = nn.Parameter(torch.Tensor(self.weight_level))
        self.activation_bias = nn.Parameter(torch.Tensor(self.activation_level))
        self.weight_clip_value = nn.Parameter(torch.Tensor([1.0]))
        self.activation_clip_value = nn.Parameter(torch.Tensor([1.0]))
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations

    def forward(self, input):
        quantized_input = quantize_activation(
            input, self.bits_activations, self.activation_clip_value, self.activation_bias.abs()
        )
        weight_mean = self.weight.data.mean()
        weight_std = self.weight.data.std()
        normalized_weight = self.weight.add(-weight_mean).div(weight_std)
        # quantized_weight = quantize_weight(
        #     normalized_weight, self.bits_weights, self.weight_clip_value.abs()
        # )
        quantized_weight = quantize_weight(
            normalized_weight, self.bits_weights, self.weight_clip_value, self.weight_bias.abs()
        )
        output = F.conv2d(
            quantized_input,
            quantized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("dorefa_clip_nonuniform2_rcf_wn_conv")
        return s
