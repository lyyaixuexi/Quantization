import copy
import math

import torch
from torch import dtype, nn
from torch.autograd import Function
from torch.nn import functional as F


# def indicator_func(x):
#     return ((x >= 0).float() - torch.sigmoid(x)).detach() + torch.sigmoid(x)

class IndicatorFunc(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.bernoulli(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SuperQConv2d(nn.Conv2d):
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
        out_channels_list=[],
    ):
        super(SuperQConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.out_channels_list = out_channels_list
        self.init_state = False
        self.output_h = 32
        self.output_w = 32

        max_out_channels = max(out_channels_list) if out_channels_list else out_channels

        channel_masks = []
        prev_out_channels = None
        for out_channels in out_channels_list:
            channel_mask = torch.ones(max_out_channels)
            channel_mask *= nn.functional.pad(torch.ones(out_channels), [0, max_out_channels - out_channels], value=0)
            if prev_out_channels:
                channel_mask *= nn.functional.pad(torch.zeros(prev_out_channels), [0, max_out_channels - prev_out_channels], value=1)
            channel_mask = channel_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            prev_out_channels = out_channels
            channel_masks.append(channel_mask)
        
        self.register_buffer('channel_masks', torch.stack(channel_masks, dim=0) if out_channels_list else None)
        self.register_parameter('choices_params', nn.Parameter(1e-3*torch.randn(len(out_channels_list) - 1)) if out_channels_list else None)

        self.bops = []
        self.indicator = None
        self.current_bops = 0
        self.probs = None

    def compute_bops(self, kernel_size, in_channels, out_channels, h, w, bits_w=32, bits_a=32):
        nk_square = in_channels * kernel_size * kernel_size
        bop = (
            out_channels
            * nk_square
            * (
                bits_w * bits_a
                + bits_w
                + bits_a
                + math.log(nk_square, 2)
            )
            * h
            * w
        )
        return bop
    
    def compute_bops_list(self):
        previous_bops = 0.0
        for index_i in range(len(self.out_channels_list)):
            output_channels = self.out_channels_list[index_i]
            current_bops = self.compute_bops(self.kernel_size[0], self.in_channels, output_channels, self.output_h, self.output_w)
            self.bops.append(current_bops - previous_bops)
            previous_bops = current_bops

    def forward(self, input):
        if not self.init_state:
            output = F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            _, _, self.output_h, self.output_w = output.shape
            self.init_state = True
            self.compute_bops_list()
        else:
            weight = self.weight
            if self.channel_masks is not None and self.choices_params is not None:
                output_mask, self.indicator, self.probs, self.current_bops = self.parametrized_mask(self.channel_masks, self.choices_params)
                weight = weight * output_mask
            output = F.conv2d(
                input,
                weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return output, self.indicator, self.probs

    def parametrized_mask(self, masks, choices_params):
        probs = torch.sigmoid(choices_params)
        if self.training:
            indicators = IndicatorFunc.apply(probs)
        else:
            indicators = (probs >= 0.5).float()
        indicators = torch.cat([indicators.new_tensor([1.0]), indicators])
        probs_ = torch.cat([probs.new_tensor([1.0]), probs])

        cum_indicators = torch.cumprod(indicators, dim=0)
        bops = indicators.new_tensor(self.bops)
        output_mask = (cum_indicators * masks).sum()
        # total_bops = (cum_indicators * bops).sum()
        total_bops = (probs_ * bops).sum()
        return output_mask, cum_indicators, probs_, total_bops

    def extra_repr(self):
        s = super().extra_repr()
        s += ", method={}".format("super_prune_bernoulii_gate_conv")
        return s


class SuperConv2d(nn.Conv2d):
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
        in_channels_list=[],
    ):
        super(SuperConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.in_channels_list = in_channels_list
        self.init_state = False
        self.output_h = 32
        self.output_w = 32

        max_in_channels = max(in_channels_list) if in_channels_list else in_channels

        channel_masks = []
        prev_in_channels = None
        for in_channels in in_channels_list:
            channel_mask = torch.ones(max_in_channels)
            channel_mask *= nn.functional.pad(torch.ones(in_channels), [0, max_in_channels - in_channels], value=0)
            if prev_in_channels:
                channel_mask *= nn.functional.pad(torch.zeros(prev_in_channels), [0, max_in_channels - prev_in_channels], value=1)
            channel_mask = channel_mask.reshape(1, channel_mask.shape[0], 1, 1)
            prev_in_channels = in_channels
            channel_masks.append(channel_mask)
        
        self.register_buffer('channel_masks', torch.stack(channel_masks, dim=0) if in_channels_list else None)

        self.bops = []
        self.current_bops = 0


    def forward(self, input, indicators, probs):
        if not self.init_state:
            output = F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            _, _, self.output_h, self.output_w = output.shape
            self.init_state = True
            self.compute_bops_list()
        else:
            weight = self.weight
            if self.channel_masks is not None:
                output_mask, self.current_bops = self.parametrized_mask(self.channel_masks, indicators, probs)
                weight = weight * output_mask
            output = F.conv2d(
                input,
                weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return output

    def parametrized_mask(self, masks, indicators, probs):
        bops = indicators.new_tensor(self.bops)
        in_masks = (indicators * masks).sum()
        # total_bops = (indicators * bops).sum()
        total_bops = (probs * bops).sum()
        return in_masks, total_bops

    def compute_bops(self, kernel_size, in_channels, out_channels, h, w, bits_w=32, bits_a=32):
        nk_square = in_channels * kernel_size * kernel_size
        bop = (
            out_channels
            * nk_square
            * (
                bits_w * bits_a
                + bits_w
                + bits_a
                + math.log(nk_square, 2)
            )
            * h
            * w
        )
        return bop
    
    def compute_bops_list(self):
        previous_bops = 0
        for index_i in range(len(self.in_channels_list)):
            in_channels = self.in_channels_list[index_i]
            current_bops = self.compute_bops(self.kernel_size[0], in_channels, self.out_channels, self.output_h, self.output_w)
            self.bops.append(current_bops - previous_bops)
            previous_bops = current_bops

    def extra_repr(self):
        s = super().extra_repr()
        s += ", method={}".format("super_gate_conv")
        return s
