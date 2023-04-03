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


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def lsq_function(v, s, q_n, q_p, k):
    if k == 32:
        return v
    s_grad_scale = 1.0 / ((q_p * v.numel()) ** 0.5)
    s_scale = grad_scale(s, s_grad_scale)

    v_bar = torch.clamp(v / s_scale, -q_n, q_p)
    v_bar = RoundFunction.apply(v_bar)
    v_hat = v_bar * s_scale
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
        self.init_state = False
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations
        self.wqn = 2 ** (self.bits_weights - 1)
        self.wqp = 2 ** (self.bits_weights - 1) - 1
        self.aqn = 0
        self.aqp = 2 ** self.bits_activations - 1
        self.w_s = nn.Parameter(data=torch.tensor(1.0))
        self.a_s = nn.Parameter(data=torch.tensor(1.0))

    def forward(self, input):
        if not self.init_state:
            self.w_s.data.fill_(self.weight.data.abs().mean() / self.wqp)
            print(
                "Weight Max:{}, magnitude s:{}, wqp: {}".format(
                    self.weight.data.max(),
                    self.w_s.data,
                    self.wqp,
                )
            )
            self.a_s.data.fill_(1.0 / self.aqp)
            print(
                "Activation Max: {}, magnitude s:{}, aqp: {}".format(
                    input.data.max(),
                    self.a_s.data,
                    self.aqp,
                )
            )
            self.init_state = True

        quantized_input = lsq_function(
            input, self.a_s, self.aqn, self.aqp, self.bits_activations
        )
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
        s += ", method={}".format("LSQ2_conv")
        return s

    def init_weight_scale(self):
        self.w_s.data.fill_(self.weight.data.abs().mean() / self.wqp)
        print(
            "Weight Max:{}, magnitude s:{}, wqp: {}".format(
                self.weight.data.max(),
                self.w_s.data,
                self.wqp,
            )
        )

    def init_activation_scale(self, input):
        self.a_s.data.fill_(1.0)
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
        self.w_s = nn.Parameter(data=torch.tensor(1.0))
        self.a_s = nn.Parameter(data=torch.tensor(1.0))

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
        s += ", method={}".format("LSQ2_linear")
        return s

    def init_weight_scale(self):
        self.w_s.data.fill_(self.weight.data.abs().mean() / self.wqp)
        print(
            "Weight Max:{}, magnitude s:{}, wqp: {}".format(
                self.weight.data.max(),
                self.w_s.data,
                self.wqp,
            )
        )

    def init_activation_scale(self, input):
        self.a_s.data.fill_(1.0)
        print(
            "Activation Max: {}, magnitude s:{}, aqp: {}".format(
                input.data.max(),
                self.a_s.data,
                self.aqp,
            )
        )

