import math
import torch

import torch.nn as nn
from torch.nn import functional as F
from compression.utils.utils import *

__all__ = ["SuperPrunedGateMemoryPreResNet", "SuperPrunedGateMemoryPreBasicBlock"]


# ---------------------------Small Data Sets Like CIFAR-10 or CIFAR-100----------------------------


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


def scale_grad(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


# both-preact | half-preact


class SuperPrunedGateMemoryPreBasicBlock(nn.Module):
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
        group_size=4
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
        super(SuperPrunedGateMemoryPreBasicBlock, self).__init__()
        self.name = block_type
        self.downsample = downsample
        self.group_size = group_size

        self.bn1 = nn.BatchNorm2d(in_plane)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(
            in_plane,
            out_plane,
            stride,
        )

        self.bn2 = nn.BatchNorm2d(out_plane)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(
            out_plane,
            out_plane,
        )

        self.n_choices = out_plane // self.group_size
        self.channel_thresholds = nn.Parameter(torch.zeros(1))
        self.register_buffer("assigned_indicator", torch.zeros(self.n_choices))
        self.conv1_bops = 0
        self.conv2_bops = 0
        self.indicator = None
        self.init_state = False
        self.input_h = []
        self.input_w = []

    def get_current_footprints(self):
        num_group = self.indicator.sum()
        num_channels = num_group * self.group_size
        n, c, h, w = self.conv1.weight.shape
        conv1_weights_footprints = compute_memory_footprint(num_channels, c, h, w)
        conv1_activations_footprints = compute_memory_footprint(1, self.conv1.in_channels, self.input_h[0], self.input_w[0])
        conv1_footprints = conv1_weights_footprints + conv1_activations_footprints

        n, c, h, w = self.conv2.weight.shape
        conv2_weights_footprints = compute_memory_footprint(n, num_channels, h, w)
        conv2_activations_footprints = compute_memory_footprint(1, num_channels, self.input_h[1], self.input_w[1])
        conv2_footprints = conv2_weights_footprints + conv2_activations_footprints

        downsample_footprints = 0
        if self.downsample is not None:
            n, c, h, w = self.downsample.weight.shape
            downsample_weights_footprints = compute_memory_footprint(n, c, h, w)
            downsample_activations_footprints = compute_memory_footprint(1, self.downsample.in_channels, self.input_h[2], self.input_w[2])
            downsample_footprints = downsample_weights_footprints + downsample_activations_footprints
        
        total_foorpints = conv1_footprints + conv2_footprints + downsample_footprints
        return total_foorpints

    def get_total_footprints(self):
        n, c, h, w = self.conv1.weight.shape
        conv1_weights_footprints = compute_memory_footprint(n, c, h, w)
        conv1_activations_footprints = compute_memory_footprint(1, self.conv1.in_channels, self.input_h[0], self.input_w[0])
        conv1_footprints = conv1_weights_footprints + conv1_activations_footprints

        n, c, h, w = self.conv2.weight.shape
        conv2_weights_footprints = compute_memory_footprint(n, c, h, w)
        conv2_activations_footprints = compute_memory_footprint(1, self.conv2.in_channels, self.input_h[1], self.input_w[1])
        conv2_footprints = conv2_weights_footprints + conv2_activations_footprints

        downsample_footprints = 0
        if self.downsample is not None:
            n, c, h, w = self.downsample.weight.shape
            downsample_weights_footprints = compute_memory_footprint(n, c, h, w)
            downsample_activations_footprints = compute_memory_footprint(1, self.downsample.in_channels, self.input_h[2], self.input_w[2])
            downsample_footprints = downsample_weights_footprints + downsample_activations_footprints
        
        total_foorpints = conv1_footprints + conv2_footprints + downsample_footprints
        return total_foorpints

    def compute_norm(self, filter_weight):
        n, c, h, w = filter_weight.shape
        filter_weight = filter_weight.reshape(n // self.group_size, self.group_size * c * h * w)
        normalized_filter_weight = filter_weight / torch.max(torch.abs(filter_weight))
        # l2-norm
        # normalized_filter_weight_norm = (normalized_filter_weight * normalized_filter_weight).sum(1) / (self.group_size * c * h * w)
        # l1-norm
        normalized_filter_weight_norm = normalized_filter_weight.abs().sum(1) / (self.group_size * c * h * w)
        # normalized_filter_weight_norm = (normalized_filter_weight.abs().sum(1) / (self.group_size * c * h * w)).detach()
        return normalized_filter_weight_norm

    def compute_indicator(self):
        # TODO: check whether to compute gradient with filter
        # TODO: check l1-norm and l2-norm
        # TODO: whether to use gradient scale, need to check on ImageNet
        # TODO: add maximum pruning bound
        filter_weight = self.conv1.weight
        normalized_filter_weight_norm = self.compute_norm(filter_weight)
        # grad_scale = 1 / math.sqrt(self.n_choices)
        # threshold = scale_grad(self.channel_thresholds, grad_scale)
        threshold = self.channel_thresholds
        self.indicator = (
            (normalized_filter_weight_norm > threshold).float()
            - torch.sigmoid(normalized_filter_weight_norm - threshold).detach()
            + torch.sigmoid(normalized_filter_weight_norm - threshold)
        )
        self.assigned_indicator.data = self.indicator.data
        return self.indicator

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
                _, _, conv1_in_shape_h, conv1_in_shape_w = x.shape
                x = self.conv1(x)
                x = self.bn2(x)
                x = self.relu2(x)
                _, _, conv2_out_shape_h, conv2_out_shape_w = x.shape
                x = self.conv2(x)
            elif self.name == "both_preact":
                residual = x
                x = self.bn1(x)
                x = self.relu1(x)
                _, _, conv1_in_shape_h, conv1_in_shape_w = x.shape
                x = self.conv1(x)
                x = self.bn2(x)
                x = self.relu2(x)
                _, _, conv2_out_shape_h, conv2_out_shape_w = x.shape
                x = self.conv2(x)

            self.input_h.append(conv1_in_shape_h)
            self.input_h.append(conv2_out_shape_h)
            self.input_w.append(conv1_in_shape_w)
            self.input_w.append(conv2_out_shape_w)

            if self.downsample:
                _, _, down_in_shape_h, down_in_shape_w = residual.shape
                residual = self.downsample(residual)
                self.input_h.append(down_in_shape_h)
                self.input_w.append(down_in_shape_w)

            out = x + residual
            self.init_state = True
        else:
            indicator = self.compute_indicator()
            if self.name == "half_preact":
                x = self.bn1(x)
                x = self.relu1(x)
                residual = x
                x = self.conv1(x)
                x = self.bn2(x)
                x = self.relu2(x)
                n, c, h, w = x.shape
                reshape_x = x.reshape(n, c // self.group_size, self.group_size, h, w)
                reshape_x = reshape_x * indicator.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x = reshape_x.reshape(n, c, h, w)
                x = self.conv2(x)
            elif self.name == "both_preact":
                residual = x
                x = self.bn1(x)
                x = self.relu1(x)
                x = self.conv1(x)
                x = self.bn2(x)
                x = self.relu2(x)
                n, c, h, w = x.shape
                reshape_x = x.reshape(n, c // self.group_size, self.group_size, h, w)
                reshape_x = reshape_x * indicator.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x = reshape_x.reshape(n, c, h, w)
                x = self.conv2(x)

            if self.downsample:
                residual = self.downsample(residual)

            out = x + residual

        return out

    def extra_repr(self):
        s = super().extra_repr()
        s += ", method={}".format("SuperPrunedGateMemoryPreBasicBlock")
        return s


class SuperPrunedGateMemoryPreResNet(nn.Module):
    """
    define SuperPreResNet on small data sets
    """

    def __init__(
        self, depth, wide_factor=1, num_classes=10, group_size=4
    ):
        """
        init model and weights
        :param depth: depth of network
        :param wide_factor: wide factor for deciding width of network, default is 1
        :param num_classes: number of classes, related to labels. default 10
        """
        super(SuperPrunedGateMemoryPreResNet, self).__init__()

        self.in_plane = 16 * wide_factor
        self.depth = depth
        n = (depth - 2) / 6
        self.conv = conv3x3(3, 16 * wide_factor)
        self.layer1 = self._make_layer(
            SuperPrunedGateMemoryPreBasicBlock,
            16 * wide_factor,
            n,
            group_size=group_size
        )
        self.layer2 = self._make_layer(
            SuperPrunedGateMemoryPreBasicBlock,
            32 * wide_factor,
            n,
            stride=2,
            group_size=group_size
        )
        self.layer3 = self._make_layer(
            SuperPrunedGateMemoryPreBasicBlock,
            64 * wide_factor,
            n,
            stride=2,
            group_size=group_size
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
        self, block, out_plane, n_blocks, stride=1, group_size=4
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
                group_size=group_size
            )
        )
        self.in_plane = out_plane
        for i in range(1, int(n_blocks)):
            layers.append(
                block(
                    self.in_plane,
                    out_plane,
                    group_size=group_size
                )
            )
        return nn.Sequential(*layers)

    def get_current_footprints(self):
        current_footprints = 0

        layer = self.conv
        n, c, h, w = layer.weight.shape
        conv_weights_footprints = compute_memory_footprint(n, c, h, w)
        conv_activations_footprints = compute_memory_footprint(1, layer.in_channels, layer.h, layer.w)
        conv_footprints = conv_weights_footprints + conv_activations_footprints
        current_footprints += conv_footprints

        for i in range(3):
            layer = getattr(self, "layer{}".format(i + 1))
            for name, module in layer.named_modules():
                if isinstance(module, SuperPrunedGateMemoryPreBasicBlock):
                    module_footprint = module.get_current_footprints()
                    current_footprints += module_footprint
        
        layer = self.fc
        n, c = layer.weight.shape
        fc_weights_footprints = compute_memory_footprint(n, c, 1, 1)
        fc_activations_footprints = compute_memory_footprint(1, layer.in_features, 1, 1)
        fc_footprint = fc_weights_footprints + fc_activations_footprints
        current_footprints += fc_footprint
        return current_footprints

    def get_total_footprints(self):
        current_footprints = 0

        layer = self.conv
        n, c, h, w = layer.weight.shape
        conv_weights_footprints = compute_memory_footprint(n, c, h, w)
        conv_activations_footprints = compute_memory_footprint(1, layer.in_channels, layer.h, layer.w)
        conv_footprints = conv_weights_footprints + conv_activations_footprints
        current_footprints += conv_footprints

        for i in range(3):
            layer = getattr(self, "layer{}".format(i + 1))
            for name, module in layer.named_modules():
                if isinstance(module, SuperPrunedGateMemoryPreBasicBlock):
                    module_footprint = module.get_total_footprints()
                    current_footprints += module_footprint
        
        layer = self.fc
        n, c = layer.weight.shape
        fc_weights_footprints = compute_memory_footprint(n, c, 1, 1)
        fc_activations_footprints = compute_memory_footprint(1, layer.in_features, 1, 1)
        fc_footprint = fc_weights_footprints + fc_activations_footprints
        current_footprints += fc_footprint
        return current_footprints

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
