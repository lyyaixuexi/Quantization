import torch
import math
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F


def step(x, b, k, left_value, right_value):
    y = torch.zeros_like(x)
    condition_1 = x - b >= 0.0
    condition_2 = x <= right_value
    condition_3 = x > left_value
    merge_condition = condition_1 & condition_2
    merge_condition_2 = ~condition_1 & condition_3
    # mask = torch.ge(x - b, 0.0)
    y[merge_condition] = k + 1.0
    y[merge_condition_2] = k
    return y


def step_backward(x, b, T, left_value, right_value):
    b_buf = x - b
    left_end_point = b - left_value
    right_end_point = right_value - b
    right_T = T / right_end_point
    left_T = T / left_end_point
    output = x.new_zeros(x.shape)
    right_condition = (b_buf >= 0) & ( x <= right_value)
    left_condition = (b_buf < 0) & (x > left_value)
    output = torch.where(right_condition, 1 / (1.0 + torch.exp(-b_buf * right_T)), output)
    output = torch.where(left_condition, 1 / (1.0 + torch.exp(-b_buf * left_T)), output)
    output = torch.where(right_condition, output * (1 - output) * right_T, output)
    output = torch.where(left_condition, output * (1 - output) * left_T, output)
    return output


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, x, b, k, T, left_value, right_value):
        self.T = T
        grad = step_backward(x, b, self.T, left_value, right_value)
        self.save_for_backward(grad)
        return step(x, b, k, left_value, right_value)

    @staticmethod
    def backward(self, grad_output):
        grad, = self.saved_tensors
        grad_input = grad * grad_output
        return grad_input, (-grad_input).sum(), None, None , None, None


def quantization(x, k, b, T):
    n = 2 ** k - 1
    # output = StepSumFunction.apply(x, b, n, T)
    scale = 1 / n
    output = 0
    for k in range(n):
        left_value = (b[k] + b[k-1]) / 2.0 if k != 0 else x.new_zeros((1,))
        right_value = (b[k] + b[k + 1]) / 2.0 if k != n - 1 else x.new_ones((1,))
        # distance = b[k] - b[]
        output += (StepFunction.apply(x, b[k], float(k), T, left_value, right_value))
    output = output * scale
    return output


def normalization_on_weights(x, clip_value):
    x = x / clip_value
    x = torch.where(x.abs() < 1, x, x.sign())
    return x


def normalization_on_activations(x, clip_value):
    x = F.relu(x)
    x = x / clip_value
    x = torch.where(x < 1, x, x.new_ones(x.shape))
    return x


def gradient_scale_function(x, scale):
    y_out = x
    y_grad = x * scale
    y = (y_out - y_grad).detach() + y_grad
    return y


def quantize_activation(x, k, clip_value, activation_bias, T):
    if k == 32:
        return x
    n = 2 ** k - 1
    grad_scale = math.sqrt(n * x.nelement())
    activation_bias = gradient_scale_function(activation_bias, 1 / grad_scale)
    x = normalization_on_activations(x, clip_value)
    x = quantization(x, k, activation_bias, T)
    x = x * clip_value
    return x


def quantize_weight(x, k, clip_value, weight_bias, T):
    if k == 32:
        return x
    n = 2 ** k - 1
    grad_scale = math.sqrt(n * x.nelement())
    weight_bias = gradient_scale_function(weight_bias, 1 / grad_scale)
    x = normalization_on_weights(x, clip_value)
    x = (x + 1.0) / 2.0
    x = quantization(x, k, weight_bias, T)
    x = x * 2.0 - 1.0
    x = x * clip_value
    return x


class RoundFunction(Function):
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
        T=1,
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
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations
        self.weight_n = int(2 ** bits_weights - 1)
        self.activation_n = int(2 ** bits_activations - 1)
        self.T = T

        self.weight_level = []
        self.activation_level = []
        self.weight_init_thrs = []
        self.activation_init_thrs = []
        if bits_weights != 32:
            for i in range(int(self.weight_n) + 1):
                self.weight_level.append(float(i) / self.weight_n)
            for i in range(int(self.weight_n)):
                self.weight_init_thrs.append(
                    (self.weight_level[i] + self.weight_level[i + 1]) / 2
                )

        if bits_activations != 32:
            for i in range(int(self.activation_n) + 1):
                self.activation_level.append(float(i) / self.activation_n)
            for i in range(int(self.activation_n)):
                self.activation_init_thrs.append(
                    (self.activation_level[i] + self.activation_level[i + 1]) / 2
                )

        # self.eps = 1e-5
        self.weight_bias = nn.Parameter(torch.Tensor(self.weight_init_thrs))
        self.activation_bias = nn.Parameter(torch.Tensor(self.activation_init_thrs))
        # self.register_buffer("weight_bias", torch.Tensor(self.weight_init_thrs))
        # self.register_buffer("activation_bias", torch.Tensor(self.activation_init_thrs))
        self.weight_clip_value = nn.Parameter(torch.Tensor([1]))
        self.activation_clip_value = nn.Parameter(torch.Tensor([1]))
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations

    def forward(self, input):
        # self.weight_bias.data, _ = torch.sort(self.weight_bias.data.abs())
        # self.activation_bias.data, _ = torch.sort(self.activation_bias.data.abs())
        self.input_nelement = input.data.nelement()
        quantized_input = quantize_activation(
            input,
            self.bits_activations,
            self.activation_clip_value.abs(),
            self.activation_bias,
            self.T,
        )
        weight_mean = self.weight.data.mean()
        weight_std = self.weight.data.std()
        normalized_weight = self.weight.add(-weight_mean).div(weight_std)
        quantized_weight = quantize_weight(
            normalized_weight,
            self.bits_weights,
            self.weight_clip_value.abs(),
            self.weight_bias,
            self.T,
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
        s += ", method={}".format("dorefa_clip_rcf_wn_conv_non_uniform")
        return s

