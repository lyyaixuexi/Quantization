import math
import torch

import torch.nn as nn
from torch.nn import functional as F
from compression.utils.utils import *
# from compression.utils.utils import compute_norm

__all__ = ["SuperSlimGatePreResNet", "SuperSlimGatePreBasicBlock"]


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


class SuperSlimGatePreBasicBlock(nn.Module):
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
        super(SuperSlimGatePreBasicBlock, self).__init__()
        self.name = block_type
        self.downsample = downsample
        self.group_size = 2

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
        self.output_h = []
        self.output_w = []

    def get_bops(self, in_channels, out_channels):
        num_group = self.indicator.sum()
        num_channels = num_group * self.group_size

        conv1_bops = compute_bops(self.conv1.kernel_size[0], in_channels, num_channels, self.output_h[0], self.output_w[0])
        conv2_bops = compute_bops(self.conv2.kernel_size[0], num_channels, out_channels, self.output_h[1], self.output_w[1])
        downsample_bops = 0
        if self.downsample is not None:
            downsample_bops = compute_bops(self.downsample.kernel_size[0], in_channels, 
                                                out_channels, self.downsample.h, self.downsample.w)
        total_bops = conv1_bops + conv2_bops + downsample_bops
        return total_bops

    def get_total_bops(self):
        conv1_bops = compute_bops(self.conv1.kernel_size[0], self.conv1.in_channels, self.conv1.out_channels, self.output_h[0], self.output_w[0])
        conv2_bops = compute_bops(self.conv2.kernel_size[0], self.conv2.in_channels, self.conv2.out_channels, self.output_h[1], self.output_w[1])
        downsample_bops = 0
        if self.downsample is not None:
            downsample_bops = compute_bops(self.downsample.kernel_size[0], self.downsample.in_channels, 
                                                self.downsample.out_channels, self.downsample.h, self.downsample.w)
        total_bops = conv1_bops + conv2_bops + downsample_bops
        return total_bops

    def compute_indicator(self):
        # TODO: check whether to compute gradient with filter
        # TODO: check l1-norm and l2-norm
        # TODO: whether to use gradient scale, need to check on ImageNet
        # TODO: add maximum pruning bound
        filter_weight = self.conv1.weight
        normalized_filter_weight_norm = compute_norm(filter_weight, self.group_size)
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
        x, current_block_indicator, previous_block_indicator = x
        if not self.init_state:
            if self.name == "half_preact":
                x = self.bn1(x)
                x = self.relu1(x)
                residual = x
                x = self.conv1(x)
                _, _, conv1_out_shape_h, conv1_out_shape_w = x.shape
                x = self.bn2(x)
                x = self.relu2(x)
                x = self.conv2(x)
                _, _, conv2_out_shape_h, conv2_out_shape_w = x.shape
            elif self.name == "both_preact":
                residual = x
                x = self.bn1(x)
                x = self.relu1(x)
                x = self.conv1(x)
                _, _, conv1_out_shape_h, conv1_out_shape_w = x.shape
                x = self.bn2(x)
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
        else:
            indicator = self.compute_indicator()
            if self.name == "half_preact":
                x = self.bn1(x)
                x = self.relu1(x)
                residual = x

                if self.downsample:
                    block_indicator = previous_block_indicator
                else:
                    block_indicator = current_block_indicator

                n, c, h, w = x.shape
                reshape_x = x.reshape(n, c // self.group_size, self.group_size, h, w)
                reshape_x = reshape_x * block_indicator.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x = reshape_x.reshape(n, c, h, w)

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

                if self.downsample:
                    block_indicator = previous_block_indicator
                else:
                    block_indicator = current_block_indicator

                n, c, h, w = x.shape
                reshape_x = x.reshape(n, c // self.group_size, self.group_size, h, w)
                reshape_x = reshape_x * block_indicator.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x = reshape_x.reshape(n, c, h, w)

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
            n, c, h, w = out.shape
            reshape_out = out.reshape(n, c // self.group_size, self.group_size, h, w)
            reshape_out = reshape_out * current_block_indicator.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out = reshape_out.reshape(n, c, h, w)

        return [out, current_block_indicator, previous_block_indicator]

    def extra_repr(self):
        s = super().extra_repr()
        s += ", method={}".format("SuperSlimGatePreBasicBlock")
        return s


class SuperSlimGatePreResNet(nn.Module):
    """
    define SuperPreResNet on small data sets
    """

    def __init__(
        self, depth, wide_factor=1, num_classes=10
    ):
        """
        init model and weights
        :param depth: depth of network
        :param wide_factor: wide factor for deciding width of network, default is 1
        :param num_classes: number of classes, related to labels. default 10
        """
        super(SuperSlimGatePreResNet, self).__init__()

        self.in_plane = 16 * wide_factor
        self.depth = depth
        n = (depth - 2) / 6
        self.conv = conv3x3(3, 16 * wide_factor)
        self.layer1 = self._make_layer(
            SuperSlimGatePreBasicBlock,
            16 * wide_factor,
            n,
        )
        self.layer2 = self._make_layer(
            SuperSlimGatePreBasicBlock,
            32 * wide_factor,
            n,
            stride=2,
        )
        self.layer3 = self._make_layer(
            SuperSlimGatePreBasicBlock,
            64 * wide_factor,
            n,
            stride=2,
        )
        self.bn = nn.BatchNorm2d(64 * wide_factor)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = linear(64 * wide_factor, num_classes)

        self.group_size = 2
        self.out_planes = [16, 32, 64]
        self.n_choices = []
        for i, out_plane in enumerate(self.out_planes):
            n_choice = out_plane // self.group_size
            self.n_choices.append(n_choice)
            setattr(self, "indicator_{}".format(i + 1), None)
            self.register_buffer("assigned_indicator_{}".format(i + 1), torch.zeros(n_choice))
        self.channel_thresholds = nn.Parameter(torch.zeros(3))
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

    def compute_indicator(self):
        for i in range(3):
            if i == 0:
                weight = self.conv.weight
            else:
                weight = getattr(self, "layer{}".format(i + 1))[0].downsample.weight
            pacesetter_norm = compute_norm(weight, self.group_size)
            threshold = self.channel_thresholds[i]
            # grad_scale = 1 / self.n_choices[i]
            # threshold = scale_grad(threshold, grad_scale)
            indicator = (
                (pacesetter_norm > threshold).float()
                - torch.sigmoid(pacesetter_norm - threshold).detach()
                + torch.sigmoid(pacesetter_norm - threshold)
            )
            setattr(self, "indicator_{}".format(i + 1), indicator)
            getattr(self, "assigned_indicator_{}".format(i + 1)).data = indicator.data

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
        self.compute_indicator()
        out = self.conv(x)

        n, c, h, w = out.shape
        reshape_out = out.reshape(n, c // self.group_size, self.group_size, h, w)
        reshape_out = reshape_out * self.indicator_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        out = reshape_out.reshape(n, c, h, w)

        out, _, _ = self.layer1([out, self.indicator_1, None])
        out, _, _ = self.layer2([out, self.indicator_2, self.indicator_1])
        out, _, _ = self.layer3([out, self.indicator_3, self.indicator_2])
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)

        n, c, h, w = out.shape
        reshape_out = out.reshape(n, c // self.group_size, self.group_size, h, w)
        reshape_out = reshape_out * self.indicator_3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        out = reshape_out.reshape(n, c, h, w)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def get_bops(self):
        num_group_1 = self.indicator_1.sum()
        num_channels_1 = num_group_1 * self.group_size

        num_group_2 = self.indicator_2.sum()
        num_channels_2 = num_group_2 * self.group_size

        num_group_3 = self.indicator_3.sum()
        num_channels_3 = num_group_3 * self.group_size

        total_bops = 0
        layer = self.conv
        bops = compute_bops(layer.kernel_size[0], layer.in_channels, num_channels_1, layer.h, layer.w)
        total_bops += bops

        in_channels = num_channels_1
        for i in range(3):
            out_channels = locals()['num_channels_{}'.format(i + 1)]
            layer = getattr(self, "layer{}".format(i + 1))
            for module in layer.modules():
                if isinstance(module, SuperSlimGatePreBasicBlock):
                    bops = module.get_bops(in_channels, out_channels)
                    total_bops += bops
                    in_channels = out_channels
        
        layer = self.fc
        bops = compute_bops(1, num_channels_3, layer.out_features, 1, 1)
        total_bops += bops
        return total_bops

    def get_total_bops(self):
        total_bops = 0
        layer = self.conv
        bops = compute_bops(layer.kernel_size[0], layer.in_channels, layer.out_channels, layer.h, layer.w)
        total_bops += bops

        for i in range(3):
            layer = getattr(self, "layer{}".format(i + 1))
            for module in layer.modules():
                if isinstance(module, (SuperSlimGatePreBasicBlock)):
                    bops = module.get_total_bops()
                    total_bops += bops
                    
        layer = self.fc
        bops = compute_bops(1, layer.in_features, layer.out_features, 1, 1)
        total_bops += bops
        return total_bops
