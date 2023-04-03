import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F


__EPS__ = 0 #1e-5


def quantization(x, k):
    n = 2 ** k - 1
    return RoundFunction.apply(x, n)


def normalization_on_weights(x):
    x = torch.tanh(x)
    x = x / torch.max(torch.abs(x)) * 0.5 + 0.5
    return x


def normalization_on_activations(x, clip_value):
    x = x / clip_value
    x = torch.clamp(x, min=0, max=1)
    return x


def quantize_activation(x, k, activation_bias):
    if k == 32:
        return x
    n = 2.0 ** k - 1.0
    sum_y = 0
    bias_sum = 0
    for i in range(int(n)):
        x_i = x - bias_sum
        y = normalization_on_activations(x_i, activation_bias[i])
        y = RoundFunction.apply(y)
        sum_y += y
        bias_sum += activation_bias[i]
    # y = y * clip_val
    return sum_y


def quantize_weight(x, k):
    if k == 32:
        return x
    x = normalization_on_weights(x)
    x = 2 * quantization(x, k) - 1
    return x


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x, n=1):
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
        self.activation_n = 2 ** bits_activations - 1
        self.activation_level = []
        if bits_activations != 32:
            for i in range(self.activation_n):
                self.activation_level.append(1.0)
        self.activation_bias = nn.Parameter(torch.Tensor(self.activation_level))
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations

    def forward(self, input):
        quantized_input = quantize_activation(
            input, self.bits_activations, self.activation_bias
        )
        # weight_mean = self.weight.data.mean()
        # weight_std = self.weight.data.std()
        # normalized_weight = self.weight.add(-weight_mean).div(weight_std)
        weight_std, weight_mean = torch.std_mean(self.weight.data.reshape(self.weight.shape[0], -1, 1, 1, 1), 1)
        normalized_weight = (self.weight - weight_mean) / (weight_std + __EPS__)
        quantized_weight = quantize_weight(normalized_weight, self.bits_weights)
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
        s += ", method={}".format("dorefa_tet_wn_nonuniform_only_activation")
        return s
