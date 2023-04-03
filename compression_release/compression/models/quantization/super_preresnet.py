import math
from numpy.lib.arraysetops import isin

import torch.nn as nn

from compression.models.quantization.super_compression_conv import SuperQConv2d, SuperCompressQConv2d
from compression.models.quantization.dorefa_clip import QConv2d, QLinear

__all__ = ["SuperPreResNet", "SuperPreBasicBlock"]


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
    1x1 quantized convolutional layer
    """
    return SuperQConv2d(
        in_plane,
        out_plane,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False,
    )


def supercconv1x1(in_plane, out_plane, stride=1):
    """
    1x1 quantized convolutional layer
    """
    return SuperCompressQConv2d(
        in_plane,
        out_plane,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False,
    )


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def qconv3x3(in_planes, out_planes, stride=1, bits_weights=32, bits_activations=32):
    "3x3 convolution with padding"
    return QConv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, bits_weights=bits_weights, bits_activations=bits_activations
    )


def superqconv3x3(in_planes, out_planes, stride=1):
    "3x3 quantized convolution with padding"
    return SuperQConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def supercconv3x3(in_planes, out_planes, stride=1):
    "3x3 quantized convolution with padding"
    return SuperCompressQConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)


# both-preact | half-preact


class SuperPreBasicBlock(nn.Module):
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
        super(SuperPreBasicBlock, self).__init__()
        self.name = block_type
        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(in_plane)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv1 = supercconv3x3(
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
        self.block_index = 0

    def forward(self, x):
        """
        forward procedure of residual module
        :param x: input feature maps
        :return: output feature maps
        """
        if self.name == "half_preact":
            x = self.bn1(x)
            x = self.relu1(x)
            residual = x
            x = self.conv1(x)
            is_tuple = False
            if isinstance(x, tuple):
                x, filter_mask = x
                is_tuple = True
            x = self.bn2(x)
            x = self.relu2(x)
            if is_tuple:
                x = self.conv2(x, filter_mask)
            else:
                x = self.conv2(x)
        elif self.name == "both_preact":
            residual = x
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv1(x)
            is_tuple = False
            if isinstance(x, tuple):
                x, filter_mask = x
                is_tuple = True
            x = self.bn2(x)
            x = self.relu2(x)
            if is_tuple:
                x = self.conv2(x, filter_mask)
            else:
                x = self.conv2(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = x + residual
        return out


class SuperPreResNet(nn.Module):
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
        super(SuperPreResNet, self).__init__()

        self.in_plane = 16 * wide_factor
        self.depth = depth
        n = (depth - 2) / 6
        self.conv = qconv3x3(3, 16 * wide_factor, bits_weights=8, bits_activations=32)
        self.layer1 = self._make_layer(
            SuperPreBasicBlock,
            16 * wide_factor,
            n,
        )
        self.layer2 = self._make_layer(
            SuperPreBasicBlock,
            32 * wide_factor,
            n,
            stride=2,
        )
        self.layer3 = self._make_layer(
            SuperPreBasicBlock,
            64 * wide_factor,
            n,
            stride=2,
        )
        self.bn = nn.BatchNorm2d(64 * wide_factor)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(8)
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
