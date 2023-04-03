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


def step(x, b):
    y = torch.zeros_like(x)
    mask = torch.ge(x - b, 0.0)
    y[mask] = 1.0
    return y


def step_backward(x, b, T, left_end_point, right_end_point):
    b_buf = x - b
    # b_output = 1 / (1.0 + torch.exp(-b_buf * T))
    # temp = b_output * (1.0 - b_output) * T
    # k = 1 / (right_end_point - left_end_point)
    left_end_point = b - left_end_point
    right_end_point = right_end_point - b
    right_T = T / right_end_point
    left_T = T / left_end_point
    output = x.new_zeros(x.shape)
    output = torch.where(b_buf >= 0, 1 / (1.0 + torch.exp(-b_buf * right_T)), output)
    output = torch.where(b_buf < 0, 1 / (1.0 + torch.exp(-b_buf * left_T)), output)
    output = torch.where(b_buf >= 0, output * (1 - output) * right_T, output)
    output = torch.where(b_buf < 0, output * (1 - output) * left_T, output)
    return output


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, x, b, T, left_end_point, right_end_point):
        self.T = T
        self.save_for_backward(x, b, left_end_point, right_end_point)
        return step(x, b)

    @staticmethod
    def backward(self, grad_output):
        x, b, left_end_point, right_end_point = self.saved_tensors
        grad = step_backward(x, b, self.T, left_end_point, right_end_point)
        grad_input = grad * grad_output
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
    output = scale * (mask + StepFunction.apply(x, B, T, left_end_point, right_end_point))
    return output


def lsq_function(v, s, q_n, q_p, k, b, T):
    if k == 32:
        return v
    v_bar = torch.clamp(v / s, -q_n, q_p)
    v_bar = quantization(v, k, b, T)
    v_hat = v_bar * s
    return v_hat


class QConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, bits_weights=32, bits_activations=32, T=1):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations
        self.wqn = 2 ** (self.bits_weights - 1)
        self.wqp = 2 ** (self.bits_weights - 1) - 1
        self.aqn = 0
        self.aqp = 2 ** self.bits_activations - 1
        self.init_state = False
        
        self.T = T
        self.weight_level = []
        self.activation_level = []
        self.weight_init_thrs = []
        self.activation_init_thrs = []
        if bits_weights != 32:
            for i in range(-self.wqn, self.wqp + 1):
                self.weight_level.append(i)
            for i in range(len(self.weight_level) - 1):
                self.weight_init_thrs.append((self.weight_level[i] + self.weight_level[i + 1]) / 2)

        if bits_activations != 32:
            for i in range(self.aqp + 1):
                self.activation_level.append(float(i))
            for i in range(len(self.activation_level) - 1):
                self.activation_init_thrs.append(
                    (self.activation_level[i] + self.activation_level[i + 1]) / 2
                )

        # self.weight_bias = nn.Parameter(torch.Tensor(self.weight_init_thrs))
        # self.activation_bias = nn.Parameter(torch.Tensor(self.activation_init_thrs))
        self.register_buffer("weight_bias", torch.Tensor(self.weight_init_thrs))
        self.register_buffer("activation_bias", torch.Tensor(self.activation_init_thrs))

        self.w_s = nn.Parameter(data=torch.tensor(2.0 * self.weight.abs().max() / math.sqrt(self.wqp)))
        self.a_s = nn.Parameter(data=torch.tensor(2.0 / math.sqrt(self.aqp)))
        # self.w_s = nn.Parameter(data=torch.tensor(self.weight.abs().max() / self.wqp))
        # self.a_s = nn.Parameter(data=torch.tensor(1.0 / self.aqp))
        # self.w_s = nn.Parameter(data=torch.FloatTensor([2 * self.weight.norm() / math.sqrt(self.wqp)]))
        # self.a_s = nn.Parameter(data=torch.FloatTensor([2 / math.sqrt(self.aqp)]))


    def forward(self, input):
        self.input_nelement = input.data.nelement() / input.data.shape[0]
        quantized_input = lsq_function(input, self.a_s, self.aqn, self.aqp, self.bits_activations, self.activation_bias, self.T)
        quantized_weight = lsq_function(self.weight, self.w_s, self.wqn, self.wqp, self.bits_weights, self.weight_bias, self.T)
        output = F.conv2d(quantized_input, quantized_weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return output

    def extra_repr(self):
        s = super().extra_repr()
        s += ', bits_weights={}'.format(self.bits_weights)
        s += ', bits_activations={}'.format(self.bits_activations)
        s += ", T={}".format(self.T)
        s += ', method={}'.format('LSQ_non_uniform')
        return s

    def init_weight_scale(self):
        # mean = self.weight.data.mean()
        # std = self.weight.data.std()
        # max_value = mean + 3 * std
        # self.w_s.data.fill_(max_value / self.wqp)
        self.w_s.data.fill_(self.weight.data.max() / self.wqp)
        # self.w_s.data = 2 * self.weight.data.norm() / math.sqrt(self.wqp)
        print('Weight Max:{}, max s:{}, magnitude s:{}, wqp: {}'.format(self.weight.data.max(), 
                                                                                            self.w_s.data, 
                                                                                            2 * self.weight.data.norm() / math.sqrt(self.wqp), self.wqp))

    def init_activation_scale(self, input):
        # self.a_s.data = input.data.max() / self.aqp
        self.a_s.data.fill_(input.data.max() * 0.25 / self.aqp)
        # self.a_s.data.fill_(2 / math.sqrt(self.aqp))
        # self.a_s.data = 2 * input.data.norm() / math.sqrt(self.aqp)
        print('Activation Max: {}, max s: {}, magnitude s:{}, aqp: {}'.format(input.data.max(), 
                                                                              self.a_s.data, 
                                                                              2 * input.data.mean() / math.sqrt(self.aqp),
                                                                              self.aqp))