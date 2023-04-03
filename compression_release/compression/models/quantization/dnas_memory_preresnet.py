import math
import torch

import torch.nn as nn
from torch.nn import functional as F
from compression.models.quantization.dorefa import QConv2d, QLinear
from compression.utils.utils import *

__all__ = ["DNASPreResNetMemory", "DNASPreBasicBlockMemory"]


# ---------------------------Small Data Sets Like CIFAR-10 or CIFAR-100----------------------------


def conv1x1(in_plane, out_plane, stride=1):
    """
    1x1 convolutional layer
    """
    return nn.Conv2d(
        in_plane, out_plane, kernel_size=1, stride=stride, padding=0, bias=False
    )


def qconv1x1(in_plane, out_plane, stride=1, bits_weights=32, bits_activations=32):
    """
    1x1 convolutional layer
    """
    return QConv2d(
        in_plane,
        out_plane,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False,
        bits_weights=bits_weights,
        bits_activations=bits_activations,
    )


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def qconv3x3(in_planes, out_planes, stride=1, bits_weights=32, bits_activations=32):
    "3x3 convolution with padding"
    return QConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        bits_weights=bits_weights,
        bits_activations=bits_activations,
    )


def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)


def build_conv3x3_candidate_ops(in_plane, out_plane, stride=1):
    candidate = []
    bits = [2, 4, 8]
    for i in bits:
        for j in bits:
            candidate.append(qconv3x3(in_plane, out_plane, stride, i, j))
    return candidate


def build_conv1x1_candidate_ops(in_plane, out_plane, stride=1):
    candidate = []
    bits = [2, 4, 8]
    for i in bits:
        for j in bits:
            candidate.append(qconv1x1(in_plane, out_plane, stride, i, j))
    return candidate


# both-preact | half-preact
class DNASPreBasicBlockMemory(nn.Module):
    """
    base module for PreResNet on small data sets
    """

    def __init__(
        self, in_plane, out_plane, stride=1, downsample=None, block_type="both_preact",
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
        super(DNASPreBasicBlockMemory, self).__init__()
        self.name = block_type
        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(in_plane)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.ModuleList(
            build_conv3x3_candidate_ops(in_plane, out_plane, stride)
        )
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.ModuleList(
            build_conv3x3_candidate_ops(out_plane, out_plane)
        )
        self.block_index = 0

        self.n_choices = 9
        # self.conv1_choices_params = nn.Parameter(torch.zeros(self.n_choices))
        # self.conv2_choices_params = nn.Parameter(torch.zeros(self.n_choices))
        # if self.downsample:
        #     self.downsample_choices_params = nn.Parameter(torch.zeros(self.n_choices))
        self.conv1_choices_params = nn.Parameter(1e-3 * torch.randn(self.n_choices))
        self.conv2_choices_params = nn.Parameter(1e-3 * torch.randn(self.n_choices))
        self.register_buffer("conv1_index", torch.FloatTensor([0]))
        self.register_buffer("conv1_bits_weights", torch.FloatTensor([2.0]))
        self.register_buffer("conv1_bits_activations", torch.FloatTensor([2.0]))
        self.register_buffer("conv2_index", torch.FloatTensor([0]))
        self.register_buffer("conv2_bits_weights", torch.FloatTensor([2.0]))
        self.register_buffer("conv2_bits_activations", torch.FloatTensor([2.0]))
        if self.downsample:
            self.downsample_choices_params = nn.Parameter(1e-3 * torch.randn(self.n_choices))
            self.register_buffer("downsample_index", torch.FloatTensor([0]))
            self.register_buffer("downsample_bits_weights", torch.FloatTensor([2.0]))
            self.register_buffer("downsample_bits_activations", torch.FloatTensor([2.0]))
        # torch.nn.init.normal_(self.choices_params, 0, 1e-3)

        self.conv1_choices_weight = None
        self.conv2_choices_weight = None
        self.downsample_choices_weight = None
        self.conv1_activation_memory_footprint = []
        self.conv1_weight_memory_footprint = []
        self.conv1_memory_footprint = []
        # self.conv1_memory_footprint = []
        self.conv2_activation_memory_footprint = []
        self.conv2_weight_memory_footprint = []
        self.conv2_memory_footprint = []
        # self.conv2_memory_footprint = []
        self.downsample_activation_memory_footprint = []
        self.downsample_weight_memory_footpront = []
        self.downsmaple_memory_footprint = []
        # self.downsample_memory_footprint = []
        self.init_state = False

    def compute_gate_output(self, choices_params, tau):
        output = F.gumbel_softmax(choices_params, tau)
        return output

    def pre_compute_memory_footprint(self):
        bits = [2, 4, 8]
        for i in bits:
            for j in bits:
                conv1_activation_memory_footprint = compute_memory_footprint(1, self.conv1[-1].c, self.conv1[-1].h, self.conv1[-1].w, j)
                conv1_weight_memory_footprint = compute_memory_footprint(self.conv1[-1].weight.shape[0], self.conv1[-1].weight.shape[1], self.conv1[-1].weight.shape[2], self.conv1[-1].weight.shape[3], i)
                self.conv1_memory_footprint.append(conv1_activation_memory_footprint + conv1_weight_memory_footprint)

                conv2_activation_memory_footprint = compute_memory_footprint(1, self.conv2[-1].c, self.conv2[-1].h, self.conv2[-1].w, j)
                conv2_weight_memory_footprint = compute_memory_footprint(self.conv2[-1].weight.shape[0], self.conv2[-1].weight.shape[1], self.conv2[-1].weight.shape[2], self.conv2[-1].weight.shape[3], i)
                self.conv2_memory_footprint.append(conv2_activation_memory_footprint + conv2_weight_memory_footprint)
                
                if self.downsample:
                    downsample_activaiton_memory_footprint = compute_memory_footprint(1, self.downsample[-1].c, self.downsample[-1].h, self.downsample[-1].w, j)
                    downsample_weight_memory_footprint = compute_memory_footprint(self.downsample[-1].weight.shape[0], self.downsample[-1].weight.shape[1], self.downsample[-1].weight.shape[2], self.downsample[-1].weight.shape[3], i)
                    self.downsmaple_memory_footprint.append(downsample_activaiton_memory_footprint + downsample_weight_memory_footprint)

    def forward(self, x):
        """
        forward procedure of residual module
        :param x: input feature maps
        :return: output feature maps
        """
        x, tau = x
        if not self.init_state:
            if self.name == "half_preact":
                x = self.bn1(x)
                x = self.relu1(x)
                residual = x
                x = self.conv1[-1](x)
                x = self.bn2(x)
                x = self.relu2(x)
                x = self.conv2[-1](x)
            elif self.name == "both_preact":
                residual = x
                x = self.bn1(x)
                x = self.relu1(x)
                x = self.conv1[-1](x)
                x = self.bn2(x)
                x = self.relu2(x)
                x = self.conv2[-1](x)

            if self.downsample:
                residual = self.downsample[-1](residual)

            out = x + residual
            self.init_state = True
            self.pre_compute_memory_footprint()
        else:
            self.conv1_choices_weight = self.compute_gate_output(self.conv1_choices_params, tau)
            self.conv2_choices_weight = self.compute_gate_output(self.conv2_choices_params, tau)
            if self.name == "half_preact":
                x = self.bn1(x)
                x = self.relu1(x)
                residual = x

                conv1_x = 0
                for i in range(self.n_choices):
                    conv1_x += self.conv1_choices_weight[i] * self.conv1[i](x)
                x = self.bn2(conv1_x)
                x = self.relu2(x)
                
                conv2_x = 0
                for i in range(self.n_choices):
                    conv2_x += self.conv2_choices_weight[i] * self.conv2[i](x)
                # x = self.conv2(x)
            elif self.name == "both_preact":
                residual = x
                x = self.bn1(x)
                x = self.relu1(x)

                conv1_x = 0
                for i in range(self.n_choices):
                    conv1_x += self.conv1_choices_weight[i] * self.conv1[i](x)
                x = self.bn2(conv1_x)
                x = self.relu2(x)

                conv2_x = 0
                for i in range(self.n_choices):
                    conv2_x += self.conv2_choices_weight[i] * self.conv2[i](x)

            residual_out = residual
            if self.downsample:
                self.downsample_choices_weight = self.compute_gate_output(self.downsample_choices_params, tau)
                residual_out = 0
                for i in range(self.n_choices):
                    residual_out += self.downsample_choices_weight[i] * self.downsample[i](residual)
                # residual = self.downsample(residual)

            out = conv2_x + residual_out

        return [out, tau]

    def compute_memory_footprint(self):
        pre_computed_conv1_memory_footprint = self.conv1[-1].weight.data.new_tensor(self.conv1_memory_footprint)
        pre_computed_conv2_memory_footprint = self.conv2[-1].weight.data.new_tensor(self.conv2_memory_footprint)
        # memory_footprint of conv1
        conv1_memory_footprint = (pre_computed_conv1_memory_footprint * self.conv1_choices_weight).sum()
        # memory_footprint of conv2
        conv2_memory_footprint = (pre_computed_conv2_memory_footprint * self.conv2_choices_weight).sum()
        downsample_memory_footprint = 0
        if self.downsample:
            pre_computed_downsample_memory_footprint = self.downsample[-1].weight.data.new_tensor(self.downsmaple_memory_footprint)
            downsample_memory_footprint = (pre_computed_downsample_memory_footprint * self.downsample_choices_weight).sum()
        total_memory_footprint = conv1_memory_footprint + conv2_memory_footprint + downsample_memory_footprint
        return total_memory_footprint

    def get_current_memory_footprint(self):
        bits = [2, 4, 8]
        max_idnex = torch.argmax(self.conv1_choices_params)
        bits_weights_index = max_idnex // 3
        bits_activation_index = max_idnex % 3
        bits_weights = bits[int(bits_weights_index)]
        bits_activation = bits[int(bits_activation_index)]
        conv1_activation_memory_footprint = compute_memory_footprint(1, self.conv1[-1].c, self.conv1[-1].h, self.conv1[-1].w, bits_activation)
        conv1_weight_memory_footprint = compute_memory_footprint(self.conv1[-1].weight.shape[0], self.conv1[-1].weight.shape[1], self.conv1[-1].weight.shape[2], self.conv1[-1].weight.shape[3], bits_weights)
        conv1_memory_footprint = conv1_activation_memory_footprint + conv1_weight_memory_footprint

        max_idnex = torch.argmax(self.conv2_choices_params)
        bits_weights_index = max_idnex // 3
        bits_activation_index = max_idnex % 3
        bits_weights = bits[int(bits_weights_index)]
        bits_activation = bits[int(bits_activation_index)]
        conv2_activation_memory_footprint = compute_memory_footprint(1, self.conv2[-1].c, self.conv2[-1].h, self.conv2[-1].w, bits_activation)
        conv2_weight_memory_footprint = compute_memory_footprint(self.conv2[-1].weight.shape[0], self.conv2[-1].weight.shape[1], self.conv2[-1].weight.shape[2], self.conv2[-1].weight.shape[3], bits_weights)
        conv2_memory_footprint = conv2_activation_memory_footprint + conv2_weight_memory_footprint

        downsample_memory_footprint = 0
        if self.downsample is not None:
            max_idnex = torch.argmax(self.downsample_choices_params)
            bits_weights_index = max_idnex // 3
            bits_activation_index = max_idnex % 3
            bits_weights = bits[int(bits_weights_index)]
            bits_activation = bits[int(bits_activation_index)]
            downsample_activaiton_memory_footprint = compute_memory_footprint(1, self.downsample[-1].c, self.downsample[-1].h, self.downsample[-1].w, bits_activation)
            downsample_weight_memory_footprint = compute_memory_footprint(self.downsample[-1].weight.shape[0], self.downsample[-1].weight.shape[1], self.downsample[-1].weight.shape[2], self.downsample[-1].weight.shape[3], bits_weights)
            downsample_memory_footprint = downsample_activaiton_memory_footprint + downsample_weight_memory_footprint
        total_memory_footprint = conv1_memory_footprint + conv2_memory_footprint + downsample_memory_footprint
        return total_memory_footprint

    def get_total_memory_footprint(self):
        conv1_activation_memory_footprint = compute_memory_footprint(1, self.conv1[-1].c, self.conv1[-1].h, self.conv1[-1].w)
        conv1_weight_memory_footprint = compute_memory_footprint(self.conv1[-1].weight.shape[0], self.conv1[-1].weight.shape[1], self.conv1[-1].weight.shape[2], self.conv1[-1].weight.shape[3])
        conv1_memory_footprint = conv1_weight_memory_footprint + conv1_activation_memory_footprint
        print("Layer: {}.{}, Weight: {}, Activation: {}".format(self.layer_name, "conv1", conv1_weight_memory_footprint, conv1_activation_memory_footprint))

        conv2_activation_memory_footprint = compute_memory_footprint(1, self.conv2[-1].c, self.conv2[-1].h, self.conv2[-1].w)
        conv2_weight_memory_footprint = compute_memory_footprint(self.conv2[-1].weight.shape[0], self.conv2[-1].weight.shape[1], self.conv2[-1].weight.shape[2], self.conv2[-1].weight.shape[3])
        conv2_memory_footprint = conv2_activation_memory_footprint + conv2_weight_memory_footprint
        print("Layer: {}.{}, Weight: {}, Activation: {}".format(self.layer_name, "conv2", conv2_weight_memory_footprint, conv2_activation_memory_footprint))

        downsample_memory_footprint = 0
        if self.downsample is not None:
            downsample_activaiton_memory_footprint = compute_memory_footprint(1, self.downsample[-1].c, self.downsample[-1].h, self.downsample[-1].w)
            downsample_weight_memory_footprint = compute_memory_footprint(self.downsample[-1].weight.shape[0], self.downsample[-1].weight.shape[1], self.downsample[-1].weight.shape[2], self.downsample[-1].weight.shape[3])
            downsample_memory_footprint = downsample_activaiton_memory_footprint + downsample_weight_memory_footprint
            print("Layer: {}.{}, Weight: {}, Activation: {}".format(self.layer_name, "downsample", downsample_weight_memory_footprint, downsample_activaiton_memory_footprint))
        total_memory_footprint = conv1_memory_footprint + conv2_memory_footprint + downsample_memory_footprint
        return total_memory_footprint

    def extra_repr(self):
        s = super().extra_repr()
        s += ", method={}".format("DNASPreBasicBlockMemory")
        return s


class DNASPreResNetMemory(nn.Module):
    """
    define SuperPreResNet on small data sets
    """

    def __init__(self, depth, wide_factor=1, num_classes=10):
        """
        init model and weights
        :param depth: depth of network
        :param wide_factor: wide factor for deciding width of network, default is 1
        :param num_classes: number of classes, related to labels. default 10
        """
        super(DNASPreResNetMemory, self).__init__()

        self.in_plane = 16 * wide_factor
        self.depth = depth
        n = (depth - 2) / 6
        # self.conv = conv3x3(3, 16 * wide_factor)
        self.conv = qconv3x3(3, 16 * wide_factor, bits_weights=8, bits_activations=32)
        self.layer1 = self._make_layer(
            DNASPreBasicBlockMemory, 16 * wide_factor, n,
        )
        self.layer2 = self._make_layer(
            DNASPreBasicBlockMemory, 32 * wide_factor, n, stride=2,
        )
        self.layer3 = self._make_layer(
            DNASPreBasicBlockMemory, 64 * wide_factor, n, stride=2,
        )
        self.bn = nn.BatchNorm2d(64 * wide_factor)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(8)
        # self.fc = linear(64 * wide_factor, num_classes)
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
            downsample = nn.ModuleList(
            build_conv1x1_candidate_ops(self.in_plane, out_plane, stride=stride)
        )

        layers = []
        layers.append(
            block(
                self.in_plane, out_plane, stride, downsample, block_type="half_preact",
            )
        )
        self.in_plane = out_plane
        for i in range(1, int(n_blocks)):
            layers.append(block(self.in_plane, out_plane,))
        return nn.Sequential(*layers)

    def forward(self, x, tau):
        """
        forward procedure of model
        :param x: input feature maps
        :return: output feature maps
        """
        out = self.conv(x)
        out, _ = self.layer1([out, tau])
        out, _ = self.layer2([out, tau])
        out, _ = self.layer3([out, tau])
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
