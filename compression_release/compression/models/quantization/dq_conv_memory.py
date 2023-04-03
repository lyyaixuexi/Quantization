import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
from compression.utils.utils import *


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class CeilFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ceil(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantize_pow2(v):
    v = torch.log2(v)
    return 2 ** RoundFunction.apply(v)


def clamp(x, min, max):
    x = torch.where(x < max, x, max)
    x = torch.where(x > min, x, min)
    return x


def quantize_activation(x, step_size, clip_value):
    x = clamp(x, min=x.new_tensor([0]), max=clip_value)
    return step_size * RoundFunction.apply(x / step_size)


def quantize_weight(x, step_size, clip_value):
    x = clamp(x, min=-clip_value, max=clip_value)
    return step_size * RoundFunction.apply(x / step_size)


class DQConv2d(nn.Conv2d):
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
    ):
        super(DQConv2d, self).__init__(
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
        self.weight_step_size = nn.Parameter(torch.Tensor([2**-3]))
        self.activation_step_size = nn.Parameter(torch.Tensor([2**-3]))
        self.weight_clip_value = nn.Parameter(torch.Tensor([1]))
        self.activation_clip_value = nn.Parameter(torch.Tensor([1]))
        self.bits_weights = 4
        self.bits_activations = 4
        self.a_s = None
        self.w_s = None

    def init_params(self):
        self.weight_step_size.data.fill_(2 ** torch.ceil(torch.log2(self.weight.abs().max() / (2 ** (self.bits_weights - 1) - 1))))
        self.weight_clip_value.data.fill_((self.weight_step_size * (2 ** (self.bits_weights - 1) - 1)).squeeze())
        # self.activation_step_size.data = 2 ** torch.round(torch.log2(self.weight.new_tensor([1.0]) / 2 ** 4 - 1))
        self.activation_clip_value.data.fill_((self.activation_step_size * (2. ** self.bits_activations - 1)).squeeze())

    def constraint_pow2(self):
        self.weight_step_size.data = quantize_pow2(self.weight_step_size).data
        self.activation_step_size.data = quantize_pow2(self.activation_step_size).data

    def forward(self, input):
        self.a_s = quantize_pow2(self.activation_step_size)
        self.w_s = quantize_pow2(self.weight_step_size)  
        quantized_input = quantize_activation(input, self.a_s, self.activation_clip_value)
        quantized_weight = quantize_weight(self.weight, self.w_s, self.weight_clip_value)

        output = F.conv2d(
            quantized_input,
            quantized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        self.output_shape = output.shape
        return output

    def compute_memory_footprint(self):
        bits_weights, bits_activations = self.compute_bits()
        activation_memory_footprint = compute_memory_footprint(1, self.c, self.h, self.w, bits_activations)
        weight_memory_footprint = compute_memory_footprint(self.weight.shape[0], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3], bits_weights)
        memory_footprint = activation_memory_footprint + weight_memory_footprint
        return memory_footprint

    def compute_bits(self):
        bits_weights = CeilFunction.apply(torch.log2(self.weight_clip_value / self.w_s + 1) + 1)
        bits_activations = CeilFunction.apply(torch.log2(self.activation_clip_value / self.a_s + 1) + 1)
        return bits_weights, bits_activations

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("dq_conv")
        return s
