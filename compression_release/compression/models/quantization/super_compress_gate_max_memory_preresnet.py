import math

import torch
import torch.nn as nn

from compression.models.quantization.dorefa_clip import QConv2d, QLinear
from compression.models.quantization.super_quan_conv_memory import SuperQConv2d
from compression.utils.utils import *

__all__ = ["SuperCompressedGatePreResNet", "SuperCompressedGatePreBasicBlockMaxMemory"]


# ---------------------------Small Data Sets Like CIFAR-10 or CIFAR-100----------------------------


def conv1x1(in_plane, out_plane, stride=1):
    """
    1x1 convolutional layer
    """
    return nn.Conv2d(
        in_plane, out_plane, kernel_size=1, stride=stride, padding=0, bias=False
    )


def superqconv1x1(in_plane, out_plane, stride=1):
    """
    1x1 convolutional layer
    """
    return SuperQConv2d(
        in_plane, out_plane, kernel_size=1, stride=stride, padding=0, bias=False
    )


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def superqconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return SuperQConv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def qconv3x3(in_planes, out_planes, stride=1, bits_weights=32, bits_activations=32):
    "3x3 convolution with padding"
    return QConv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, bits_weights=bits_weights, bits_activations=bits_activations
    )


def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)


def scale_grad(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


# both-preact | half-preact


class SuperCompressedGatePreBasicBlockMaxMemory(nn.Module):
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
        super(SuperCompressedGatePreBasicBlockMaxMemory, self).__init__()
        self.name = block_type
        self.downsample = downsample
        self.group_size = group_size

        self.bn1 = nn.BatchNorm2d(in_plane)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = superqconv3x3(
            in_plane,
            out_plane,
            stride,
        )

        self.bn2 = nn.BatchNorm2d(out_plane)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = superqconv3x3(
            out_plane,
            out_plane,
        )

        self.n_choices = out_plane // self.group_size
        self.channel_thresholds = nn.Parameter(torch.zeros(1))
        self.register_buffer("assigned_indicator", torch.zeros(self.n_choices))
        self.indicator = None
        self.init_state = False
        self.output_h = []
        self.output_w = []
        self.output_n = []
        self.output_c = []

    def get_downsample_memory_footprint(self):
        downsample_memory_footprint = 0
        if self.downsample is not None:
            downsample_activation_memory_footprint = compute_memory_footprint(self.output_n[2], self.output_c[2], self.output_h[2], self.output_w[2])
            downsample_weight_memory_footprint = compute_memory_footprint(self.downsample.weight.shape[0], self.downsample.weight.shape[1], self.downsample.weight.shape[2], self.downsample.weight.shape[3])
            downsample_memory_footprint = downsample_activation_memory_footprint + downsample_weight_memory_footprint
        return downsample_memory_footprint
    
    def compress_fix_activation_compute_weight_memory_footprint(self):
        num_group = self.indicator.sum()
        channel_num = num_group * self.group_size
        # memory_footprint of conv1
        conv1_memory_footprint = self.conv1.compress_fix_activation_compute_weight_memory_footprint(channel_num, is_out_channel=True)
        # memory_footprint of conv2
        conv2_memory_footprint = self.conv2.compress_fix_activation_compute_weight_memory_footprint(channel_num, is_out_channel=False)
        # memory_footprint of downsample     
        # downsample_memory_footprint = self.get_downsample_memory_footprint()
        downsample_memory_footprint = 0
        if self.downsample is not None:
            downsample_memory_footprint = self.downsample.fix_activation_compute_weight_memory_footprint()
        total_memory_footprint = conv1_memory_footprint + conv2_memory_footprint + downsample_memory_footprint
        return total_memory_footprint

    def compress_fix_weight_compute_activation_memory_footprint(self):
        num_group = self.indicator.sum()
        channel_num = num_group * self.group_size
        # memory_footprint of conv1
        conv1_memory_footprint = self.conv1.compress_fix_weight_compute_activation_memory_footprint(channel_num, is_out_channel=True)
        # memory_footprint of conv2
        conv2_memory_footprint = self.conv2.compress_fix_weight_compute_activation_memory_footprint(channel_num, is_out_channel=False)
        # downsample_memory_footprint = self.get_downsample_memory_footprint()
        downsample_memory_footprint = 0
        if self.downsample is not None:
            downsample_memory_footprint = self.downsample.fix_weight_compute_activation_memory_footprint()
        total_memory_footprint = conv1_memory_footprint + conv2_memory_footprint + downsample_memory_footprint
        return total_memory_footprint

    def compute_memory_footprint(self):
        num_group = self.indicator.sum()
        channel_num = num_group * self.group_size
        # memory_footprint of conv1
        conv1_memory_footprint = self.conv1.compute_current_memory_footprint(channel_num, is_out_channel=True)
        # memory_footprint of conv2
        conv2_memory_footprint = self.conv2.compute_current_memory_footprint(channel_num, is_out_channel=False)
        downsample_memory_footprint = self.get_downsample_memory_footprint()
        total_memory_footprint = conv1_memory_footprint + conv2_memory_footprint + downsample_memory_footprint
        return total_memory_footprint

    def get_total_memory_footprint(self):
        conv1_activation_memory_footprint = compute_memory_footprint(self.output_n[0], self.output_c[0], self.output_h[0], self.output_w[0])
        conv1_weight_memory_footprint = compute_memory_footprint(self.conv1.weight.shape[0], self.conv1.weight.shape[1], self.conv1.weight.shape[2], self.conv1.weight.shape[3])
        conv1_memory_footprint = conv1_activation_memory_footprint + conv1_weight_memory_footprint

        conv2_activation_memory_footprint = compute_memory_footprint(self.output_n[1], self.output_c[1], self.output_h[1], self.output_w[1])
        conv2_weight_memory_footprint = compute_memory_footprint(self.conv2.weight.shape[0], self.conv2.weight.shape[1], self.conv2.weight.shape[2], self.conv2.weight.shape[3])
        conv2_memory_footprint = conv2_activation_memory_footprint + conv2_weight_memory_footprint

        downsample_activation_memory_footprint = compute_memory_footprint(self.output_n[2], self.output_c[2], self.output_h[2], self.output_w[2])
        downsample_weight_memory_footprint = compute_memory_footprint(self.downsample.weight.shape[0], self.downsample.weight.shape[1], self.downsample.weight.shape[2], self.downsample.weight.shape[3])
        downsample_memory_footprint = downsample_activation_memory_footprint + downsample_weight_memory_footprint
        total_memory_footprint = conv1_memory_footprint + conv2_memory_footprint + downsample_memory_footprint
        return total_memory_footprint

    def compute_norm(self, filter_weight):
        n, c, h, w = filter_weight.shape
        filter_weight = filter_weight.reshape(n // self.group_size, self.group_size * c * h * w)
        # range [-1, 1]
        normalized_filter_weight = filter_weight / torch.max(torch.abs(filter_weight))
        # range [0, 1]
        # normalized_filter_weight = filter_weight / torch.max(torch.abs(filter_weight)) * 0.5 + 0.5
        # normalized_filter_weight = filter_weight
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
        # TODO: whether to use quantized filter?
        filter_weight = self.conv1.weight
        # quantized_weight = normalize_and_quantize_weight(filter_weight, self.conv1.bits_weights, self.conv1.weight_clip_value.detach())
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
                x = self.conv1(x)
                conv1_out_shape_n, conv1_out_shape_c, conv1_out_shape_h, conv1_out_shape_w = x.shape
                x = self.bn2(x)
                x = self.relu2(x)
                x = self.conv2(x)
                conv2_out_shape_n, conv2_out_shape_c, conv2_out_shape_h, conv2_out_shape_w = x.shape
            elif self.name == "both_preact":
                residual = x
                x = self.bn1(x)
                x = self.relu1(x)
                x = self.conv1(x)
                conv1_out_shape_n, conv1_out_shape_c, conv1_out_shape_h, conv1_out_shape_w = x.shape
                x = self.bn2(x)
                x = self.relu2(x)
                x = self.conv2(x)
                conv2_out_shape_n, conv2_out_shape_c, conv2_out_shape_h, conv2_out_shape_w = x.shape

            if self.downsample:
                residual = self.downsample(residual)
                residual_out_shape_n, residual_out_shape_c, residual_out_shape_h, residual_out_shape_w = residual.shape
            self.output_n.append(conv1_out_shape_n)
            self.output_n.append(conv2_out_shape_n)
            self.output_n.append(residual_out_shape_n)
            self.output_c.append(conv1_out_shape_c)
            self.output_c.append(conv2_out_shape_c)
            self.output_c.append(residual_out_shape_c)
            self.output_h.append(conv1_out_shape_h)
            self.output_h.append(conv2_out_shape_h)
            self.output_h.append(residual_out_shape_h)
            self.output_w.append(conv1_out_shape_w)
            self.output_w.append(conv2_out_shape_w)
            self.output_w.append(residual_out_shape_w)

            out = x + residual
            self.init_state = True
            # self.compute_base_memory_footprint()
        else:
            indicator = self.compute_indicator()
            if self.name == "half_preact":
                x = self.bn1(x)
                x = self.relu1(x)
                residual = x

                n, _, _, _ = self.conv1.weight.shape
                indicator = indicator.reshape(-1, 1)
                indicator = indicator.expand(n // self.group_size, self.group_size).reshape(n)
                index = (indicator > 0).nonzero().squeeze()
                selected_weight = torch.index_select(self.conv1.weight, 0, index)
                weight_mean = selected_weight.data.mean()
                weight_std = selected_weight.data.std()

                x = self.conv1(x, self.conv1.weight, weight_mean, weight_std)
                x = self.bn2(x)
                x = self.relu2(x)

                selected_weight = torch.index_select(self.conv2.weight, 1, index)
                weight_mean = selected_weight.data.mean()
                weight_std = selected_weight.data.std()
                x = x * indicator.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

                x = self.conv2(x, self.conv2.weight, weight_mean, weight_std)
            elif self.name == "both_preact":
                residual = x
                x = self.bn1(x)
                x = self.relu1(x)

                n, _, _, _ = self.conv1.weight.shape
                indicator = indicator.reshape(-1, 1)
                indicator = indicator.expand(n // self.group_size, self.group_size).reshape(n)
                index = (indicator > 0).nonzero().squeeze()
                selected_weight = torch.index_select(self.conv1.weight, 0, index)
                weight_mean = selected_weight.data.mean()
                weight_std = selected_weight.data.std()

                x = self.conv1(x, self.conv1.weight, weight_mean, weight_std)
                x = self.bn2(x)
                x = self.relu2(x)

                selected_weight = torch.index_select(self.conv2.weight, 1, index)
                weight_mean = selected_weight.data.mean()
                weight_std = selected_weight.data.std()
                x = x * indicator.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

                x = self.conv2(x, self.conv2.weight, weight_mean, weight_std)

            if self.downsample:
                residual = self.downsample(residual)

            out = x + residual

        return out

    def extra_repr(self):
        s = super().extra_repr()
        s += ", method={}".format("SuperCompressedGatePreBasicBlock")
        return s


class SuperCompressedGatePreResNetMaxMemory(nn.Module):
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
        super(SuperCompressedGatePreResNetMaxMemory, self).__init__()

        self.in_plane = 16 * wide_factor
        self.depth = depth
        n = (depth - 2) / 6
        # self.conv = superqweightconv3x3(3, 16 * wide_factor)
        self.conv = qconv3x3(3, 16 * wide_factor, bits_weights=8, bits_activations=32)
        self.layer1 = self._make_layer(
            SuperCompressedGatePreBasicBlockMaxMemory,
            16 * wide_factor,
            n,
            group_size=group_size
        )
        self.layer2 = self._make_layer(
            SuperCompressedGatePreBasicBlockMaxMemory,
            32 * wide_factor,
            n,
            stride=2,
            group_size=group_size
        )
        self.layer3 = self._make_layer(
            SuperCompressedGatePreBasicBlockMaxMemory,
            64 * wide_factor,
            n,
            stride=2,
            group_size=group_size
        )
        self.bn = nn.BatchNorm2d(64 * wide_factor)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(8)
        # self.fc = SuperQLinear(64 * wide_factor, num_classes)
        self.fc = QLinear(64 * wide_factor, num_classes, bits_weights=8, bits_activations=8)

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
            downsample = superqconv1x1(
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
