from torch.nn.modules import activation
import math
import torch

import torch.nn as nn
from torch.nn import functional as F

from compression.models.quantization.super_prune_conv_bernoulli_gate import SuperQConv2d, SuperConv2d

__all__ = ["SuperPrunedPreResNet", "SuperPrunedPreBasicBlock"]


# ---------------------------Small Data Sets Like CIFAR-10 or CIFAR-100----------------------------

ratios = [0.6, 0.7, 0.8, 0.9, 1.0]


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


def conv1x1(in_plane, out_plane, stride=1):
    """
    1x1 convolutional layer
    """
    return nn.Conv2d(
        in_plane, out_plane, kernel_size=1, stride=stride, padding=0, bias=False
    )


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)


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


# both-preact | half-preact


class SuperPrunedPreBasicBlock(nn.Module):
    """
    base module for PreResNet on small data sets
    """

    def __init__(
        self,
        in_plane,
        out_plane,
        stride=1,
        downsample=None,
        block_type="both_preact",
    ):
        """
        init module and weights
        :param in_plane: size of input plane
        :param out_plane: size of output plane
        :param stride: stride of convolutional layers, default 1
        :param downsample: down sample type for expand dimension of input feature maps, default None
        :param block_type: type of blocks, decide position of short cut, both-preact: short cut start from beginning
        of the first segment, half-preact: short cut start from the position between the first segment and the second
        one. default: both-preact
        """
        super(SuperPrunedPreBasicBlock, self).__init__()
        self.name = block_type
        self.downsample = downsample
        self.out_channels_list = []
        for ratio in ratios:
            self.out_channels_list.append(int(out_plane * ratio))
        self.out_channels_list = sorted( list( set(self.out_channels_list) ) )

        self.bn1 = nn.BatchNorm2d(in_plane)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(
            in_plane,
            out_plane,
            stride,
        )
        # self.bn2 = nn.BatchNorm2d(out_plane)
        self.bn2s = nn.ModuleList()
        for _out in self.out_channels_list:
            self.bn2s.append(nn.BatchNorm2d(_out))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(
            out_plane,
            out_plane,
        )

        max_out_channels = max(self.out_channels_list) if self.out_channels_list else out_plane       
        channel_masks = []
        for out_channels in self.out_channels_list:
            channel_mask = torch.ones(max_out_channels)
            channel_mask *= nn.functional.pad(torch.ones(out_channels), [0, max_out_channels - out_channels], value=0)
            channel_mask = channel_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            channel_masks.append(channel_mask)

        self.n_choices = len(self.out_channels_list)
        self.choices_params = nn.Parameter(torch.zeros(self.n_choices))
        self.choices_path_weight = nn.Parameter(torch.zeros(self.n_choices))
        self.register_buffer('channel_masks', torch.stack(channel_masks, dim=0))
        # torch.nn.init.normal_(self.choices_params, 0, 1e-3)

        self.bops = [[], []]
        self.output_h = []
        self.output_w = []
        self.active_index = [0]
        self.inactive_index = None

        self.log_prob = None
        self.current_prob_over_ops = None
        self.init_state = False
        self.MODE = None

    def entropy(self, eps=1e-8):
        probs = self.current_prob_over_ops
        log_probs = torch.log(probs + eps)
        entropy = - torch.sum(torch.mul(probs, log_probs))
        return entropy

    def set_chosen_op_active(self):
        probs = F.softmax(self.choices_params, dim=0).data
        chosen_idx = torch.argmax(probs)
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
        for index_i in range(self.n_choices):
            output_channels = self.out_channels_list[index_i]
            conv1_bops = self.compute_bops(self.conv1.kernel_size[0], self.conv1.in_channels, output_channels, self.output_h[0], self.output_w[0])
            conv2_bops = self.compute_bops(self.conv2.kernel_size[0], output_channels, self.conv2.out_channels, self.output_h[1], self.output_w[1])
            self.bops[0].append(conv1_bops)
            self.bops[1].append(conv2_bops)

    def binarize(self):
        self.choices_path_weight.data.zero_()
        probs = F.softmax(self.choices_params, dim=0)
        sample = torch.multinomial(probs.data, 1)[0].item()
        self.active_index = [sample]
        self.inactive_index = [_i for _i in range(0, sample)] + \
                                  [_i for _i in range(sample + 1, self.n_choices)]
        self.log_prob = torch.log(probs[sample])
        self.current_prob_over_ops = probs
        # set binary gate
        self.choices_path_weight.data[sample] = 1.0
        for name, param in self.named_parameters():
            param.grad = None

    def set_arch_param_grad(self):
        binary_grads = self.choices_path_weight.grad.data
        if self.choices_params.grad is None:
            self.choices_params.grad = torch.zeros_like(self.choices_params.data)
        probs = self.current_prob_over_ops.data
        for i in range(self.n_choices):
            for j in range(self.n_choices):
                self.choices_params.grad.data[i] += binary_grads[j] * probs[j] * (delta_ij(i, j) - probs[i])

    def path_index_forward(self, x_, active_id):
        # single_x = x_
        # channel_num = self.out_channels_list[active_id]
        # selected_weight = self.conv2.weight[:, :channel_num]
        channel_num = self.out_channels_list[active_id]
        selected_x = x_[:, :channel_num]
        out_x = self.relu2(self.bn2s[active_id](selected_x))
        selected_weight = self.conv2.weight[:, :channel_num]

        # active_masks = self.channel_masks[active_id]
        # masked_weight = self.conv2.weight * active_masks
        bias = self.conv2.bias
        stride = self.conv2.stride
        padding = self.conv2.padding
        dilation = self.conv2.dilation
        groups = self.conv2.groups
        output = F.conv2d(
            out_x,
            selected_weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )
        return output

    def get_bops(self):
        bops = self.conv1.weight.new_tensor(self.bops)
        conv1_bops = (self.current_prob_over_ops * bops[0]).sum()
        conv2_bops = (self.current_prob_over_ops * bops[1]).sum()
        downsample_bops = 0
        if self.downsample is not None:
            downsample_bops = self.compute_bops(self.downsample.kernel_size[0], self.downsample.in_channels, 
                                                self.downsample.out_channels, self.downsample.h, self.downsample.w)
        total_bops = conv1_bops + conv2_bops + downsample_bops
        return total_bops

    def forward(self, x):
        """
        forward procedure of residual module
        :param x: input feature maps
        :return: output feature maps
        """
        if not self.init_state:
            if self.name == "half_preact":
                x = self.bn1(x)
                x = self.relu1(x)
                residual = x
                x = self.conv1(x)
                _, _, conv1_out_shape_h, conv1_out_shape_w = x.shape
                x = self.bn2s[-1](x)
                # x = self.bn2(x)
                x = self.relu2(x)
                x = self.conv2(x)
                _, _, conv2_out_shape_h, conv2_out_shape_w = x.shape
            elif self.name == "both_preact":
                residual = x
                x = self.bn1(x)
                x = self.relu1(x)
                x = self.conv1(x)
                _, _, conv1_out_shape_h, conv1_out_shape_w = x.shape
                x = self.bn2s[-1](x)
                # x = self.bn2(x)
                x = self.relu2(x)
                x = self.conv2(x)
                _, _, conv2_out_shape_h, conv2_out_shape_w = x.shape

            if self.downsample:
                residual = self.downsample(residual)
            self.output_h.append(conv1_out_shape_h)
            self.output_h.append(conv2_out_shape_h)
            self.output_w.append(conv1_out_shape_w)
            self.output_w.append(conv2_out_shape_w)

            out = x + residual
            self.init_state = True
            self.compute_bops_list()
        else:
            # self.binarize()
            if self.MODE == "full":
                def run_function(active_id):
                    def forward(_x):
                        return self.path_index_forward(_x, active_id)
                    return forward

                def backward_function(active_id, binary_gates):
                    def backward(_x, _output, grad_output):
                        binary_grads = torch.zeros_like(binary_gates.data)
                        with torch.no_grad():
                            for k in range(self.n_choices):
                                if k != active_id:
                                    out_k = self.path_index_forward(_x.data, k)
                                else:
                                    out_k = _output.data
                                grad_k = torch.sum(out_k * grad_output)
                                binary_grads[k] = grad_k
                        return binary_grads
                    return backward

                if self.name == "half_preact":
                    x = self.bn1(x)
                    x = self.relu1(x)
                    residual = x
                    x = self.conv1(x)
                    # x = self.bn2(x)
                    # x = self.relu2(x)
                    # out_x = [x[:,:out_channels] for out_channels in self.out_channels_list]
                    # out_relu = [self.relu2(self.bn2s[idx](out_conv)) for idx, out_conv in enumerate(out_x)]

                    x = ArchGradientFunction.apply(
                        x, self.choices_path_weight, run_function(self.active_index[0]),
                        backward_function(self.active_index[0], self.choices_path_weight))
                elif self.name == "both_preact":
                    residual = x
                    x = self.bn1(x)
                    x = self.relu1(x)
                    x = self.conv1(x)
                    # x = self.bn2(x)
                    # x = self.relu2(x)
                    # out_x = [x[:,:out_channels] for out_channels in self.out_channels_list]
                    # out_relu = [self.relu2(self.bn2s[idx](out_conv)) for idx, out_conv in enumerate(out_x)]

                    x = ArchGradientFunction.apply(
                        x, self.choices_path_weight, run_function(self.active_index[0]),
                        backward_function(self.active_index[0], self.choices_path_weight))
                if self.downsample:
                    residual = self.downsample(residual)

                out = x + residual      
            else:
                if self.name == "half_preact":
                    x = self.bn1(x)
                    x = self.relu1(x)
                    residual = x
                    x = self.conv1(x)
                    # x = self.bn2(x)
                    # x = self.relu2(x)
                    # out_x = [x[:,:out_channels] for out_channels in self.out_channels_list]
                    # out_relu = [self.relu2(self.bn2s[idx](out_conv)) for idx, out_conv in enumerate(out_x)]
                    x = self.path_index_forward(x, self.active_index[0])

                elif self.name == "both_preact":
                    residual = x
                    x = self.bn1(x)
                    x = self.relu1(x)
                    x = self.conv1(x)
                    # x = self.bn2(x)
                    # x = self.relu2(x)
                    # out_x = [x[:,:out_channels] for out_channels in self.out_channels_list]
                    # out_relu = [self.relu2(self.bn2s[idx](out_conv)) for idx, out_conv in enumerate(out_x)]
                    x = self.path_index_forward(x, self.active_index[0])

                if self.downsample:
                    residual = self.downsample(residual)

                out = x + residual

        return out

    def extra_repr(self):
        s = super().extra_repr()
        s += ", method={}".format("SuperPrunedPreBasicBlock")
        return s


class SuperPrunedPreResNet(nn.Module):
    """
    define SuperPreResNet on small data sets
    """

    def __init__(
        self, depth, quantize_first_last=False, wide_factor=1, num_classes=10
    ):
        """
        init model and weights
        :param depth: depth of network
        :param wide_factor: wide factor for deciding width of network, default is 1
        :param num_classes: number of classes, related to labels. default 10
        """
        super(SuperPrunedPreResNet, self).__init__()

        self.in_plane = 16 * wide_factor
        self.depth = depth
        n = (depth - 2) / 6
        self.conv = conv3x3(3, 16 * wide_factor)
        self.layer1 = self._make_layer(
            SuperPrunedPreBasicBlock,
            16 * wide_factor,
            n,
        )
        self.layer2 = self._make_layer(
            SuperPrunedPreBasicBlock,
            32 * wide_factor,
            n,
            stride=2,
        )
        self.layer3 = self._make_layer(
            SuperPrunedPreBasicBlock,
            64 * wide_factor,
            n,
            stride=2,
        )
        self.bn = nn.BatchNorm2d(64 * wide_factor)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = linear(64 * wide_factor, num_classes)

        self._init_weight()

    def _init_weight(self):
        # init layer parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()

    def _make_layer(
        self, block, out_plane, n_blocks, stride=1
    ):
        """
        make residual blocks, including short cut and residual function
        :param block: type of basic block to build network
        :param out_plane: size of output plane
        :param n_blocks: number of blocks on every segment
        :param stride: stride of convolutional neural network, default 1
        :return: residual blocks
        """
        downsample = None
        if stride != 1 or self.in_plane != out_plane:
            downsample = conv1x1(
                self.in_plane,
                out_plane,
                stride=stride,
            )

        layers = []
        layers.append(
            block(
                self.in_plane,
                out_plane,
                stride,
                downsample,
                block_type="half_preact",
            )
        )
        self.in_plane = out_plane
        for i in range(1, int(n_blocks)):
            layers.append(
                block(
                    self.in_plane,
                    out_plane,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward procedure of model
        :param x: input feature maps
        :return: output feature maps
        """
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
