import torch
import math
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, v):
        return torch.round(v)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class LSQFunction(Function):
    @staticmethod
    def forward(ctx, v, s, q_n, q_p):
        ctx.save_for_backward(v, s)
        ctx.other = q_n, q_p
        v_bar = torch.clamp(v / s, -q_n, q_p)
        v_bar = torch.round(v_bar)
        v_hat = v_bar * s
        return v_hat

    @staticmethod
    def backward(ctx, grad_output):
        v, s = ctx.saved_tensors
        q_n, q_p = ctx.other
        v_div_s = v / s

        condition_1 = v > -q_n
        condition_2 = v < q_p
        merge_condition = condition_1 & condition_2
        grad_zero = grad_output.new_zeros(grad_output.shape)
        grad_v = torch.where(merge_condition, grad_output, grad_zero)

        v_grad_s = v.new_full(v.shape, q_p)
        q_n_tensor = v.new_full(v.shape, -q_n)
        condition_1 = v_div_s <= -q_n
        condition_2 = v_div_s >= q_p
        merge_condition = (~condition_1) & (~condition_2)
        v_grad_s = torch.where(condition_1, q_n_tensor, v_grad_s)
        v_grad_s = torch.where(
            merge_condition, -v_div_s + torch.round(v_div_s), v_grad_s
        )

        grad_scale = math.sqrt(v.nelement() * q_p)
        v_grad_s = v_grad_s / grad_scale
        return grad_v, (grad_output * v_grad_s).sum(), None, None


def lsq_function(v, s, q_n, q_p, k):
    if k == 32:
        return v
    v_bar = torch.clamp(v / s, -q_n, q_p)
    v_bar = RoundFunction.apply(v_bar)
    v_hat = v_bar * s
    return v_hat


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
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations
        self.wqn = 2 ** (self.bits_weights - 1)
        self.wqp = 2 ** (self.bits_weights - 1) - 1
        self.aqn = 0
        self.aqp = 2 ** self.bits_activations - 1
        self.w_s = nn.Parameter(data=torch.tensor(1.0 / self.wqp))
        self.a_s = nn.Parameter(data=torch.tensor(1.0 / self.aqp))

    def forward(self, input):
        # self.input_nelement = input.data.nelement() / input.data.shape[0]
        self.input_nelement = input.data.nelement()
        # quantized_input = LSQFunction.apply(input, self.a_s, self.aqn, self.aqp)
        quantized_input = lsq_function(
            input, self.a_s, self.aqn, self.aqp, self.bits_activations
        )
        # weight_mean = self.weight.data.mean()
        # weight_std = self.weight.data.std()
        # normalized_weight = self.weight.add(-weight_mean).div(weight_std)
        # quantized_weight = LSQFunction.apply(self.weight, self.w_s, self.wqn, self.wqp)
        quantized_weight = lsq_function(
            self.weight, self.w_s, self.wqn, self.wqp, self.bits_weights
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
        s += ", method={}".format("LSQ_conv")
        return s

    def init_weight_scale(self):
        # mean = self.weight.data.mean()
        # std = self.weight.data.std()
        # max_value = mean + 3 * std
        # self.w_s.data.fill_(max_value / self.wqp)
        # self.w_s.data.fill_(self.weight.data.max() / self.wqp)
        self.w_s.data.fill_(2 * self.weight.data.abs().mean() / math.sqrt(self.wqp))
        # self.w_s.data = 2 * self.weight.data.norm() / math.sqrt(self.wqp)
        print(
            "Weight Max:{}, magnitude s:{}, wqp: {}".format(
                self.weight.data.max(),
                self.w_s.data,
                self.wqp,
            )
        )

    def init_activation_scale(self, input):
        # self.a_s.data = input.data.max() / self.aqp
        # self.a_s.data.fill_(input.data.max() * 0.25 / self.aqp)
        # self.a_s.data.fill_(2 / math.sqrt(self.aqp))
        # self.a_s.data = 2 * input.data.norm() / math.sqrt(self.aqp)
        self.a_s.data.fill_(2 * input.data.abs().mean() / math.sqrt(self.aqp))
        print(
            "Activation Max: {}, magnitude s:{}, aqp: {}".format(
                input.data.max(),
                self.a_s.data,
                self.aqp,
            )
        )


class QLinear(nn.Linear):
    """
    custom convolutional layers for quantization
    """

    def __init__(
        self,
        in_features, 
        out_features, 
        bias=True,
        bits_weights=32,
        bits_activations=32,
    ):
        super(QLinear, self).__init__(
            in_features, 
            out_features, 
            bias=True
        )
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations
        self.wqn = 2 ** (self.bits_weights - 1)
        self.wqp = 2 ** (self.bits_weights - 1) - 1
        self.aqn = 0
        self.aqp = 2 ** self.bits_activations - 1
        self.w_s = nn.Parameter(data=torch.tensor(1.0 / self.wqp))
        self.a_s = nn.Parameter(data=torch.tensor(1.0 / self.aqp))

    def forward(self, input):
        self.input_nelement = input.data.nelement()
        quantized_input = lsq_function(
            input, self.a_s, self.aqn, self.aqp, self.bits_activations
        )
        quantized_weight = lsq_function(
            self.weight, self.w_s, self.wqn, self.wqp, self.bits_weights
        )
        output = F.linear(quantized_input, quantized_weight, self.bias)
        return output

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("LSQ_linear")
        return s

    def init_weight_scale(self):
        self.w_s.data.fill_(2 * self.weight.data.abs().mean() / math.sqrt(self.wqp))
        print(
            "Weight Max:{}, magnitude s:{}, wqp: {}".format(
                self.weight.data.max(),
                self.w_s.data,
                self.wqp,
            )
        )

    def init_activation_scale(self, input):
        self.a_s.data.fill_(2 * input.data.abs().mean() / math.sqrt(self.aqp))
        print(
            "Activation Max: {}, magnitude s:{}, aqp: {}".format(
                input.data.max(),
                self.a_s.data,
                self.aqp,
            )
        )

