import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F


def step(x, b):
    y = torch.zeros_like(x)
    mask = torch.ge(x - b, 0.0)
    y[mask] = 1.0
    return y


def step_backward(x, b, T, left_end_point, right_end_point, n):
    b_buf = x - b
    left_end_point = b - left_end_point
    right_end_point = right_end_point - b
    output = x.new_zeros(x.shape)
    output = torch.where(b_buf >= 0, 0.5 / right_end_point, output)
    output = torch.where(b_buf < 0, 0.5 / left_end_point, output)
    # output = x.new_zeros(x.shape)
    # output.data.fill_(n)
    return output


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, x, b, T, left_end_point, right_end_point, n):
        self.T = T
        self.n = n
        self.save_for_backward(x, b, left_end_point, right_end_point)
        return step(x, b)

    @staticmethod
    def backward(self, grad_output):
        x, b, left_end_point, right_end_point = self.saved_tensors
        grad = step_backward(x, b, self.T, left_end_point, right_end_point, self.n)
        grad_input = grad * grad_output
        return grad_input, -grad_input, None, None, None, None


def quantization(x, k, b, T):
    n = 2 ** k - 1
    scale = 1 / n

    # with torch.no_grad():
        # a = torch.

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
    output = scale * (
        mask + StepFunction.apply(x, B, T, left_end_point, right_end_point, n)
    )
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


def quantize_activation(x, k, clip_value, activation_bias, T):
    if k == 32:
        return x
    x = normalization_on_activations(x, clip_value)
    x = quantization(x, k, activation_bias, T)
    x = x * clip_value
    return x


def quantize_weight(x, k, clip_value, weight_bias, T):
    if k == 32:
        return x
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
                self.weight_init_thrs.append(
                    (self.weight_level[i] + self.weight_level[i + 1]) / 2
                )

        if bits_activations != 32:
            for i in range(self.activation_n + 1):
                self.activation_level.append(float(i) / self.activation_n)
            for i in range(self.activation_n):
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
        # self.weight_bias.data, _ = torch.sort(self.weight_bias.data)
        # self.activation_bias.data, _ = torch.sort(self.activation_bias.data)
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
        s += ", method={}".format("dorefa_clip_line_rcf_wn_conv")
        return s