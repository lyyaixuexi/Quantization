import torch
import math
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F


def step(x, b):
    return (x >= b).float()


def step_backward_sigmoid(x, b, endpoints_T):
    b_buf = x - b
    # output = 1 / (1.0 + torch.exp(-b_buf * endpoints_T))
    output = torch.sigmoid(b_buf * endpoints_T)
    output = output * (1 - output) * endpoints_T
    return output


def step_backward_line(x, n):
    return x.new_tensor(n)


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, x, b, endpoints_T, n):
        grad_sigmoid = step_backward_sigmoid(x, b, endpoints_T)
        grad_line = step_backward_line(x, n)
        self.save_for_backward(grad_sigmoid, grad_line)
        return step(x, b)

    @staticmethod
    def backward(self, grad_output):
        grad_sigmoid, grad_line = self.saved_tensors
        grad_input = grad_sigmoid * grad_output
        return grad_line * grad_output, -grad_input, None, None


def quantization(x, k, b, T):
    n = 2 ** k - 1
    scale = 1 / n

    with torch.no_grad():
        b1 = b[1:]
        b2 = b[0:-1]
        interval_endpoints = torch.cat([b.new_tensor([0.0]), (b1 + b2) / 2.0, b.new_tensor([1.0])])
        x_shape = x.shape
        unsqueeze_x = x.unsqueeze(-1)
        nelement = unsqueeze_x.nelement()

        # shape: (n, 1)
        interval_index = ((unsqueeze_x > interval_endpoints).long().sum(-1) - 1).reshape(-1, 1)
        interval_index = torch.clamp(interval_index, min=0)

        b_interval_endpoints = torch.cat([b, b.new_tensor([1.0])])
        b_endpoints_distance = torch.cat([b - interval_endpoints[:-1], (interval_endpoints[-1] - b[-1]).reshape(1)])
        
        # shape: (nelement, n)
        expand_endpoints_T = (T / b_endpoints_distance).unsqueeze(0).expand(nelement, -1)
        # shape: (n, 1)
        endpoints_T_index = (unsqueeze_x >= b_interval_endpoints).long().sum(-1).reshape(-1, 1)
        endpoints_T_index = torch.clamp(endpoints_T_index, max=n)
        endpoints_T = torch.gather(expand_endpoints_T, 1, endpoints_T_index).reshape(x_shape)
    
    # shape: (nelement, n)
    expand_b = b.unsqueeze(0).expand(nelement, -1)
    B = torch.gather(expand_b, 1, interval_index).reshape(x_shape)
    interval_index = interval_index.reshape(x_shape).float()
    output = scale * (
        interval_index + StepFunction.apply(x, B, endpoints_T, n)
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
        s += ", method={}".format("dorefa_clip_rcf_wn_conv_non_uniform_sigmoid_line")
        return s


class QLinear(nn.Linear):
    """
    custom convolutional layers for quantization
    """

    def __init__(
        self, in_features, out_features, bias=True, bits_weights=32, bits_activations=32
    ):
        super(QLinear, self).__init__(in_features, out_features, bias=bias)
        # self.eps = 1e-5
        self.clip_value = nn.Parameter(torch.Tensor([1]))
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations

    def forward(self, input):
        quantized_input = quantize_activation(
            input, self.bits_activations, self.clip_value
        )
        weight_mean = self.weight.data.mean()
        weight_std = self.weight.data.std()
        normalized_weight = self.weight.add(-weight_mean).div(weight_std)
        quantized_weight = quantize_weight(normalized_weight, self.bits_weights)
        output = F.linear(quantized_input, quantized_weight, self.bias)
        return output

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("dorefa_rcf_wn_linear")
        return s
