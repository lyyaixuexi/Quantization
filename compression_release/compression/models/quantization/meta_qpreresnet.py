import math

import torch.nn as nn
# from compression.models.quantization.dorefa import QConv2d
# from compression.models.quantization.dorefa_wn import QConv2d
# from compression.models.quantization.dorefa_wn_old import QConv2d
from compression.models.quantization.meta_dorefa_wn import QConv2d

__all__ = ["MetaQPreResNet", "MetaQPreBasicBlock"]

bits = [4, 2]

# ---------------------------Small Data Sets Like CIFAR-10 or CIFAR-100----------------------------


def conv1x1(in_plane, out_plane, stride=1):
    """
    1x1 convolutional layer
    """
    return nn.Conv2d(in_plane, out_plane, kernel_size=1, stride=stride, padding=0, bias=False)


def qconv1x1(in_plane, out_plane, stride=1):
    """
    1x1 quantized convolutional layer
    """
    return QConv2d(in_plane, out_plane, kernel_size=1, 
                   stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def qconv3x3(in_planes, out_planes, stride=1):
    "3x3 quantized convolution with padding"
    return QConv2d(in_planes, out_planes, kernel_size=3, 
                   stride=stride, padding=1, bias=False)


def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)


# both-preact | half-preact


class MetaQPreBasicBlock(nn.Module):
    """
    base module for PreResNet on small data sets
    """

    def __init__(self, in_plane, out_plane, stride=1, downsample=None, block_type="both_preact"):
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
        super(MetaQPreBasicBlock, self).__init__()
        self.name = block_type
        self.downsample = downsample

        self.bn1 = nn.ModuleList([])
        for bit in bits:
            self.bn1.append(nn.BatchNorm2d(in_plane))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = qconv3x3(in_plane, out_plane, stride)
        self.bn2 = nn.ModuleList([])
        for bit in bits:
            self.bn2.append(nn.BatchNorm2d(out_plane))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = qconv3x3(out_plane, out_plane)
        self.block_index = 0

    def forward(self, x, bit_id):
        """
        forward procedure of residual module
        :param x: input feature maps
        :return: output feature maps
        """
        if self.name == "half_preact":
            x = self.bn1[bit_id](x)
            x = self.relu1(x)
            residual = x
            x = self.conv1(x, bit_id)
            x = self.bn2[bit_id](x)
            x = self.relu2(x)
            x = self.conv2(x, bit_id)
        elif self.name == "both_preact":
            residual = x
            x = self.bn1[bit_id](x)
            x = self.relu1(x)
            x = self.conv1(x, bit_id)
            x = self.bn2[bit_id](x)
            x = self.relu2(x)
            x = self.conv2(x, bit_id)

        if self.downsample:
            residual = self.downsample(residual, bit_id)

        out = x + residual
        return out


class MetaQPreResNet(nn.Module):
    """
    define QPreResNet on small data sets
    """

    def __init__(self, depth, wide_factor=1, num_classes=10):
        """
        init model and weights
        :param depth: depth of network
        :param wide_factor: wide factor for deciding width of network, default is 1
        :param num_classes: number of classes, related to labels. default 10
        """
        super(MetaQPreResNet, self).__init__()

        self.in_plane = 16 * wide_factor
        self.depth = depth
        n = (depth - 2) / 6
        self.conv = conv3x3(3, 16 * wide_factor)
        self.layer1 = self._make_layer(MetaQPreBasicBlock, 16 * wide_factor, n)
        self.layer2 = self._make_layer(MetaQPreBasicBlock, 32 * wide_factor, n, stride=2)
        self.layer3 = self._make_layer(MetaQPreBasicBlock, 64 * wide_factor, n, stride=2)
        self.bn = nn.ModuleList([])
        for bit in bits:
            self.bn.append(nn.BatchNorm2d(64 * wide_factor))
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

    def _make_layer(self, block, out_plane, n_blocks, stride=1):
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
            downsample = qconv1x1(self.in_plane, out_plane, stride=stride)

        layers = []
        layers.append(block(self.in_plane, out_plane, stride, 
                            downsample, block_type="half_preact"))
        self.in_plane = out_plane
        for i in range(1, int(n_blocks)):
            layers.append(block(self.in_plane, out_plane))
        return nn.Sequential(*layers)

    def forward(self, x, bit_id):
        """
        forward procedure of model
        :param x: input feature maps
        :return: output feature maps
        """
        out = self.conv(x)
        for i, block in enumerate(self.layer1):
            out = block(out, bit_id)
        # out = self.layer1(out, bit_id)
        for i, block in enumerate(self.layer2):
            out = block(out, bit_id)
        # out = self.layer2(out, bit_id)
        for i, block in enumerate(self.layer3):
            out = block(out, bit_id)
        # out = self.layer3(out, bit_id)
        out = self.bn[bit_id](out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
