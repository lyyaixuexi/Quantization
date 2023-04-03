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
    x = x * clip_value
    x = torch.clamp(x, min=0, max=1)
    return x


def normalization_on_weights_negative(x, clip_value):
    x = x * clip_value
    x = torch.clamp(x, min=-1.0, max=0.0)
    return x


def normalization_on_weights_positive(x, clip_value):
    x = x * clip_value
    x = torch.clamp(x, min=0.0, max=1.0)
    return x


def quantize_activation(x, k, activation_bias, activation_clip):
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
        bias_sum += 1.0/activation_bias[i]
    sum_y = sum_y / n
    y = y * activation_clip
    return sum_y


# def quantize_weight(x, k):
#     if k == 32:
#         return x
#     x = normalization_on_weights(x)
#     x = 2 * quantization(x, k) - 1
#     return x


def quantize_weight(x, k, weight_bias, weight_clip):
    if k == 32:
        return x
    x = normalization_on_weights(x)
    x = 2 * quantize_activation(x, k, weight_bias, weight_clip) - 1
    x = x * weight_clip
    return x


# def quantize_weight(x, k, weight_bias):
#     if k == 32:
#         return x
#     n = 2.0 ** k - 1.0
#     qn = 2.0 ** (k - 1)
#     qp = 2.0 ** (k - 1) - 1
#     sum_y = 0
#     # neg_bias_sum = 0
#     # pos_bias_sum = 0
#     # for i in range(int(qn)):
#     #     # x - (- bias_sum)
#     #     x_i = x + neg_bias_sum
#     #     y = normalization_on_weights_negative(x_i, weight_bias[i])
#     #     y = RoundFunction.apply(y)
#     #     sum_y += y
#     #     neg_bias_sum += 1.0/weight_bias[i]

#     # for i in range(int(qp)):
#     #     x_i = x - pos_bias_sum
#     #     bias = weight_bias[i + int(qn)]
#     #     y = normalization_on_weights_positive(x_i, bias)
#     #     y = RoundFunction.apply(y)
#     #     sum_y += y
#     #     pos_bias_sum += 1.0/bias

#     # y1 = x * weight_bias[0]
#     # y1 = torch.clamp(y1, min=-1, max=0)
#     # y1 = RoundFunction.apply(y1)
#     # y2 = x * weight_bias[1]
#     # y2 = torch.clamp(y2, min=0, max=1)
#     # y2 = RoundFunction.apply(y2)
#     # y = y1 + y2
#     # return sum_y
#     return y


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
        self.init_state = False
        self.weight_n = 2 ** bits_weights - 1
        self.activation_n = 2 ** bits_activations - 1
        self.weight_level = []
        self.activation_level = []
        if bits_weights != 32:
            for i in range(self.weight_n):
                self.weight_level.append(self.weight_n)
            # for i in range(2):
            #     self.weight_level.append(1.0)
        if bits_activations != 32:
            for i in range(self.activation_n):
                self.activation_level.append(1.0)
        # self.register_buffer("weight_bias", torch.Tensor(self.weight_level))
        self.weight_bias = nn.Parameter(torch.Tensor(self.weight_level))
        self.activation_bias = nn.Parameter(torch.Tensor(self.activation_level))
        self.weight_clip = nn.Parameter(torch.Tensor([1.0]))
        self.activation_clip = nn.Parameter(torch.Tensor([1.0]))
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations

    def forward(self, input):
        quantized_input = quantize_activation(
            input, self.bits_activations, self.activation_bias, self.activation_clip
        )
        # weight_mean = self.weight.data.mean()
        # weight_std = self.weight.data.std()
        # normalized_weight = self.weight.add(-weight_mean).div(weight_std)
        weight_std, weight_mean = torch.std_mean(self.weight.data.reshape(self.weight.shape[0], -1, 1, 1, 1), 1)
        normalized_weight = (self.weight - weight_mean) / (weight_std + __EPS__)
        # print(self.weight_bias)
        quantized_weight = quantize_weight(normalized_weight, self.bits_weights, self.weight_bias, self.weight_clip)
        # quantized_weight = quantize_weight(normalized_weight, self.bits_weights)
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
        s += ", method={}".format("dorefa_tet_wn_nonuniform_2")
        return s
