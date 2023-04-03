import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F


class SignFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, x, b, left_endpoint, right_endpoint):
        self.save_for_backward(x, b, left_endpoint, right_endpoint)
        return torch.sign(x - b)

    @staticmethod
    def backward(self, grad_output):
        x, b, left_endpoint, right_endpoint = self.saved_tensors
        grad_input = x.new_zeros(x.shape)
        left_endpoint = left_endpoint - b
        right_endpoint = right_endpoint - b
        x = x - b
        # grad_input = torch.where(x > right_endpoint, x.new_tensor(0.0), grad_input)
        # grad_input = torch.where(x < left_endpoint, x.new_tensor(0.0), grad_input)
        mask_pos = (x >= 0.0) & (x < right_endpoint)
        mask_neg = (x < 0.0) & (x >= left_endpoint)
        grad_input = torch.where(mask_pos, -2 * x / (right_endpoint * right_endpoint) + 2 / right_endpoint, grad_input)
        grad_input = torch.where(mask_neg, 2 * x / (left_endpoint * left_endpoint) - 2 / left_endpoint, grad_input)
        # grad = step_backward(x, b, self.T, left_endpoint, right_endpoint)
        grad_input = grad_input * grad_output
        return grad_input, -grad_input, None, None, None


def quantization(x, k, b, T):
    n = 2 ** k - 1
    scale = 1 / n
    mask = x.new_zeros(x.shape)
    interval_endpoints = []
    interval_endpoints.append(x.new_tensor(0.0))
    for i in range(n - 1):
        interval_endpoint = (b[i] + b[i + 1]) / 2.0
        interval_endpoints.append(interval_endpoint)
        mask = torch.where(x > interval_endpoint, x.new_tensor([i + 1]), mask)
    interval_endpoints.append(x.new_tensor(1.0))
    interval_endpoints = torch.stack(interval_endpoints, dim=0).reshape(-1)

    # mask shape: (nelement, 1)
    reshape_mask = mask.reshape(-1, 1).long()
    nelement = reshape_mask.shape[0]
    # expand_b shape: (nelement, n)
    expand_b = b.unsqueeze(0).expand(nelement, n)
    # expand_interval_endpoints shape: (nelement, -1)
    expand_interval_endpoints = interval_endpoints.unsqueeze(0).expand(nelement, -1)

    # B shape: (nelement)
    B = torch.gather(expand_b, 1, reshape_mask)
    left_end_point = torch.gather(expand_interval_endpoints, 1, reshape_mask)
    right_end_point = torch.gather(expand_interval_endpoints, 1, reshape_mask + 1)
    B = B.reshape(x.shape)
    left_end_point = left_end_point.reshape(x.shape)
    right_end_point = right_end_point.reshape(x.shape)
    output = scale * (mask + SignFunction.apply(x, B, left_end_point, right_end_point) / 2.0 + 0.5)
    return output


def normalization_on_weights(x):
    x = torch.tanh(x)
    x = x / torch.max(torch.abs(x)) * 0.5 + 0.5
    return x


def normalization_on_activations(x, clip_value):
    x = F.relu(x)
    x = x / clip_value
    x = torch.where(x < 1, x, x.new_ones(x.shape))
    return x


def quantize_activation(x, k, clip_value, activation_bias, T):
    if k == 32:
        return x
    x = normalization_on_activations(x, clip_value)
    x = quantization(x, k, activation_bias, T)
    x = x * clip_value
    return x


def quantize_weight(x, k, weight_bias, T):
    if k == 32:
        return x
    x = normalization_on_weights(x)
    x = 2 * quantization(x, k, weight_bias, T) - 1
    return x


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
        T=1
    ):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        # self.eps = 1e-5
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations
        self.weight_n = 2 ** bits_weights - 1
        self.activation_n = 2 ** bits_activations - 1
        self.T = T

        self.weight_level = []
        self.activation_level = []
        self.weight_init_thrs = []
        self.activation_init_thrs = []
        if bits_weights != 32:
            for i in range(self.weight_n + 1):
                self.weight_level.append(float(i) / self.weight_n)
            for i in range(self.weight_n):
                self.weight_init_thrs.append((self.weight_level[i] + self.weight_level[i + 1]) / 2)

        if bits_activations != 32:
            for i in range(self.activation_n + 1):
                self.activation_level.append(float(i) / self.activation_n)
            for i in range(self.activation_n):
                self.activation_init_thrs.append(
                    (self.activation_level[i] + self.activation_level[i + 1]) / 2
                )

        self.weight_bias = nn.Parameter(torch.Tensor(self.weight_init_thrs))
        self.activation_bias = nn.Parameter(torch.Tensor(self.activation_init_thrs))
        # self.register_buffer("weight_bias", torch.Tensor(self.weight_init_thrs))
        # self.register_buffer("activation_bias", torch.Tensor(self.activation_init_thrs))
        self.clip_value = nn.Parameter(torch.Tensor([1]))

    def forward(self, input):
        self.input_nelement = input.data.nelement()
        quantized_input = quantize_activation(
            input, self.bits_activations, self.clip_value, self.activation_bias, self.T)
        weight_mean = self.weight.data.mean()
        weight_std = self.weight.data.std()
        normalized_weight = self.weight.add(-weight_mean).div(weight_std)
        # normalized_weight = self.weight
        quantized_weight = quantize_weight(
            normalized_weight, self.bits_weights, self.weight_bias, self.T)
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
        s += ", T={}".format(self.T)
        s += ", method={}".format("dorefa_wn_non_uniform_parabola")
        return s
