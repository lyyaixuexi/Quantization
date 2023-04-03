from compression.models.quantization.test_non_uniform import output
import copy
import math
import numpy as np

import torch
from torch import dtype, nn
from torch.autograd import Function
from torch.nn import functional as F

def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0

def detach_variable(inputs):
    if isinstance(inputs, tuple):
        return tuple([detach_variable(x) for x in inputs])
    else:
        x = inputs.detach()
        x.requires_grad = inputs.requires_grad
        return x

# def indicator_func(x):
#     return ((x >= 0).float() - torch.sigmoid(x)).detach() + torch.sigmoid(x)

class IndicatorFunc(Function):
    @staticmethod
    def forward(ctx, probs):
        return torch.multinomial(probs, 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ArchGradientFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, binary_gates, run_func, backward_func):
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        detached_x = detach_variable(x)
        with torch.enable_grad():
            output = run_func(detached_x)
        ctx.save_for_backward(detached_x, output)
        return output.data

    @staticmethod
    def backward(ctx, grad_output):
        detached_x, output = ctx.saved_tensors

        grad_x = torch.autograd.grad(output, detached_x, grad_output, only_inputs=True)
        # compute gradients w.r.t. binary_gates
        binary_grads = ctx.backward_func(detached_x.data, output.data, grad_output.data)

        return grad_x[0], binary_grads, None, None


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
        self.n_chocies = len(self.out_channels_list)
        self.MODE = None

        max_out_channels = max(out_channels_list) if out_channels_list else out_channels

        channel_masks = []
        for out_channels in out_channels_list:
            channel_mask = torch.ones(max_out_channels)
            channel_mask *= nn.functional.pad(torch.ones(out_channels), [0, max_out_channels - out_channels], value=0)
            channel_mask = channel_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            channel_masks.append(channel_mask)
        
        self.register_buffer('channel_masks', torch.stack(channel_masks, dim=0) if out_channels_list else None)
        self.channels_path_weight = nn.Parameter(torch.Tensor(self.n_chocies))
        self.choices_params = nn.Parameter(torch.Tensor(self.n_chocies))

        self.bops = []
        self.current_bops = 0

        self.active_index = [0]
        self.inactive_index = None

        self.log_prob = None
        self.current_prob_over_ops = None

    @property
    def n_choices(self):
        return len(self.out_channels_list)

    @property
    def probs_over_ops(self):
        probs = F.softmax(self.choices_params, dim=0)  # softmax to probability
        return probs

    @property
    def chosen_index(self):
        probs = self.probs_over_ops.data
        index = torch.argmax(probs)
        return index, probs[index]

    def entropy(self, eps=1e-8):
        probs = self.probs_over_ops
        log_probs = torch.log(probs + eps)
        entropy = - torch.sum(torch.mul(probs, log_probs))
        return entropy

    def set_chosen_op_active(self):
        chosen_idx, _ = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(chosen_idx + 1, self.n_choices)]

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
        for index_i in range(len(self.out_channels_list)):
            output_channels = self.out_channels_list[index_i]
            current_bops = self.compute_bops(self.kernel_size[0], self.in_channels, output_channels, self.output_h, self.output_w)
            self.bops.append(current_bops)

    def path_index_forward(self, x_, active_id):
        active_masks = self.channel_masks[active_id]
        masked_weight = self.weight * active_masks
        return F.conv2d(
            x_,
            masked_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

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
            if self.MODE == "full":
                # self.compute_path_weight()
                self.binarize()
                def run_function(active_id):
                    def forward(_x):
                        self.path_index_forward(_x, active_id)
                    return forward

                def backward_function(active_id, binary_gates):
                    def backward(_x, _output, grad_output):
                        binary_grads = torch.zeros_like(binary_gates.data)
                        with torch.no_grad():
                            for k in range(self.n_chocies):
                                if k != active_id:
                                    out_k = self.path_index_forward(_x.data, k)
                                else:
                                    out_k = _output.data
                                grad_k = torch.sum(out_k * grad_output)
                                binary_grads[k] = grad_k
                        return binary_grads
                    return backward
                output = ArchGradientFunction.apply(
                    input, self.compute_path_weight, run_function(self.active_index[0]),
                    backward_function(self.active_index[0], self.compute_path_weight))
            else:
                output = self.path_index_forward(input, self.active_index[0])
        return output, self.active_index, self.probs

    def set_chosen_op_active(self):
        chosen_idx, _ = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(chosen_idx + 1, self.n_choices)]

    def binarize(self):
        self.channels_path_weight.data.zero_()
        probs = F.softmax(self.choices_params, dim=0)
        sample = torch.multinomial(probs.data, 1)[0].item()
        self.active_index = [sample]
        self.inactive_index = [_i for _i in range(0, sample)] + \
                                  [_i for _i in range(sample + 1, self.n_choices)]
        self.log_prob = torch.log(probs[sample])
        self.current_prob_over_ops = probs
        # set binary gate
        self.channels_path_weight.data[sample] = 1.0

    def set_arch_param_grad(self):
        binary_grads = self.channels_path_weight.grad.data
        if self.choices_params.grad is None:
            self.choices_params.grad = torch.zeros_like(self.choices_params.data)
        probs = self.current_prob_over_ops.data
        for i in range(self.n_choices):
            for j in range(self.n_choices):
                self.choices_params.grad.data[i] += binary_grads[j] * probs[j] * (delta_ij(i, j) - probs[i])


    def get_total_bops(self):
        probs = self.current_prob_over_ops
        bops = probs.new_tensor(self.bops)
        total_bops = (bops * probs).sum()
        return total_bops
    # def parametrized_mask(self, masks, choices_params):
    #     probs = F.softmax(choices_params, dim=0)
    #     sample_index = IndicatorFunc.apply(probs)

    #     output_mask = masks[sample_index]
    #     # output_mask = torch.index_select(masks, 0, sample_index).squeeze(dim=0)
    #     bops = masks.new_tensor(self.bops)
    #     total_bops = bops[sample_index]
    #     # total_bops = torch.index_select(bops, 0, sample_index)
    #     # total_bops = (probs * bops).sum()

    #     return output_mask, sample_index, probs, total_bops

    def extra_repr(self):
        s = super().extra_repr()
        s += ", method={}".format("super_prune_multinomial_gate_conv")
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
        for in_channels in in_channels_list:
            channel_mask = torch.ones(max_in_channels)
            channel_mask *= nn.functional.pad(torch.ones(in_channels), [0, max_in_channels - in_channels], value=0)
            channel_mask = channel_mask.reshape(1, channel_mask.shape[0], 1, 1)
            channel_masks.append(channel_mask)
        
        self.register_buffer('channel_masks', torch.stack(channel_masks, dim=0) if in_channels_list else None)

        self.bops = []
        self.current_bops = 0


    def forward(self, input, sample_index, probs):
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
                output_mask, self.current_bops = self.parametrized_mask(self.channel_masks, sample_index, probs)
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

    def parametrized_mask(self, masks, sample_index, probs):
        in_masks = torch.index_select(masks, 0, sample_index).squeeze(dim=0)
        bops = masks.new_tensor(self.bops)
        # total_bops = torch.index_select(bops, 0, sample_index)
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
        for index_i in range(len(self.in_channels_list)):
            in_channels = self.in_channels_list[index_i]
            current_bops = self.compute_bops(self.kernel_size[0], in_channels, self.out_channels, self.output_h, self.output_w)
            self.bops.append(current_bops)

    def extra_repr(self):
        s = super().extra_repr()
        s += ", method={}".format("super_gate_conv")
        return s
