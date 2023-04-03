import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F

bits = [4, 2]

def quantization(x, k):
    n = 2 ** k - 1
    return RoundFunction.apply(x, n)


def normalization_on_weights(x):
    x = torch.tanh(x)
    x = x / torch.max(torch.abs(x)) * 0.5 + 0.5
    return x


def normalization_on_activations(x, clip_value):
    x = F.relu(x)
    x = x / clip_value
    x = torch.where(x < 1, x, x.new_ones(x.shape))
    return x


def quantize_activation(x, k, clip_value):
    if k == 32:
        return x
    x = normalization_on_activations(x, clip_value)
    x = quantization(x, k)
    x = x * clip_value
    return x


def quantize_weight(x, k):
    if k == 32:
        return x
    x = normalization_on_weights(x)
    x = 2 * quantization(x, k) - 1
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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)
        self.eps = 1e-5

        self.adaptor = nn.ModuleList([])
        # self.bn = []
        self.clip_value = nn.ParameterList([])
        for i, bit in enumerate(bits):
            if i != 0:
                # self.adaptor.append(nn.Conv2d(in_channels, in_channels, 1, bias=False))
                self.adaptor.append(nn.Linear(kernel_size * kernel_size, kernel_size * kernel_size))
            # self.bn.append(nn.BatchNorm2d(in_planes))
            self.clip_value.append(nn.Parameter(torch.Tensor([6])))

    def forward(self, input, bit_id):
        quantized_input = quantize_activation(input, bits[bit_id], self.clip_value[bit_id])

        # get weight
        # full_precision_weight = self.adaptor[bit_id](self.weight)
        if bit_id == 0:
            full_precision_weight = self.weight
        else:
            n, c, h, w = self.weight.shape
            self.weight_reshape = self.weight.reshape(n, c, h * w)
            full_precision_weight = self.adaptor[bit_id - 1](self.weight_reshape)
            full_precision_weight = full_precision_weight.reshape(n, c, h, w)

        # normalize weight
        weight_mean = full_precision_weight.data.mean()
        weight_std = full_precision_weight.data.std()
        normalized_weight = full_precision_weight.add(-weight_mean).div(weight_std + self.eps)

        quantized_weight = quantize_weight(normalized_weight, bits[bit_id])
        output = F.conv2d(quantized_input, quantized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return output

    def extra_repr(self):
        s = super().extra_repr()
        s += ', bits_weights={}'.format(bits)
        s += ', bits_activations={}'.format(bits)
        s += ', method={}'.format('meta_dorefa_rcf_wn')
        return s
