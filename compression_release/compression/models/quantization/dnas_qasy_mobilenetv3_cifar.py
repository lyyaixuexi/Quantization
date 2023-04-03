"""
MobileNetV3 From <Searching for MobileNetV3>, arXiv:1905.02244.
Ref: https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
     https://github.com/kuan-wang/pytorch-mobilenet-v3/blob/master/mobilenetv3.py
     https://github.com/ShowLo/MobileNetV3
     
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from compression.models.quantization.dorefa_clip_asymmetric import QConv2d, QLinear
from compression.utils.utils import *


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3.0, self.inplace) / 6.0
        return out * x


def _make_divisible(v, divisor=8, min_value=None):
    """
    Ensure that 'number' can be 'divisor' divisible
    Reference from original tensorflow repo:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def build_conv_candidate_ops(
    in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True
):
    candidate = []
    bits = [2, 4, 8]
    for i in bits:
        for j in bits:
            candidate.append(
                QConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                    bits_weights=i,
                    bits_activations=j,
                )
            )
    return candidate


def build_fc_candidate_ops(in_features, out_features):
    candidate = []
    bits = [2, 4, 8]
    for i in bits:
        for j in bits:
            candidate.append(
                QLinear(in_features, out_features, bits_weights=i, bits_activations=j)
            )
    return candidate


class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.ModuleList(build_fc_candidate_ops(exp_size, exp_size // divide)),
            nn.ReLU(inplace=True),
            nn.ModuleList(build_fc_candidate_ops(exp_size // divide, exp_size)),
            h_sigmoid(),
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        # out = self.dense(out)
        out = self.dense[0][-1](out)
        out = self.dense[1](out)
        out = self.dense[2][-1](out)
        out = self.dense[3](out)
        out = out.view(batch, channels, 1, 1)
        return out * x


class DNASAsyMobileBlockCifar(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernal_size, stride, nonLinear, SE, exp_size,
    ):
        super(DNASAsyMobileBlockCifar, self).__init__()
        self.out_channels = out_channels
        self.nonLinear = nonLinear
        self.SE = SE
        self.expand = exp_size != in_channels
        padding = (kernal_size - 1) // 2

        if self.nonLinear == "RE":
            activation = nn.ReLU
        else:
            activation = h_swish

        if self.expand:
            self.conv1 = nn.ModuleList(
                build_conv_candidate_ops(
                    in_channels,
                    exp_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            )
            self.bn1 = nn.BatchNorm2d(exp_size)
            self.relu1 = activation(inplace=True)

        self.conv2 = nn.ModuleList(
            build_conv_candidate_ops(
                exp_size,
                exp_size,
                kernel_size=kernal_size,
                stride=stride,
                padding=padding,
                groups=exp_size,
                bias=False,
            )
        )
        self.bn2 = nn.BatchNorm2d(exp_size)
        self.relu2 = activation(inplace=True)

        self.conv3 = nn.ModuleList(
            build_conv_candidate_ops(
                exp_size, out_channels, kernel_size=1, stride=1, padding=0, bias=False
            )
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)

        self.use_connect = stride == 1 and in_channels == out_channels

        self.n_choices = 9
        if self.expand:
            self.conv1_choices_params = nn.Parameter(1e-3 * torch.randn(self.n_choices))
            self.register_buffer("conv1_index", torch.FloatTensor([8]))
            self.register_buffer("conv1_bits_weights", torch.FloatTensor([8.0]))
            self.register_buffer("conv1_bits_activations", torch.FloatTensor([8.0]))
            self.conv1_choices_weight = None
            self.conv1_bops = []
        self.conv2_choices_params = nn.Parameter(1e-3 * torch.randn(self.n_choices))
        self.conv3_choices_params = nn.Parameter(1e-3 * torch.randn(self.n_choices))

        if self.SE:
            self.se_fc1_choices_params = nn.Parameter(
                1e-3 * torch.randn(self.n_choices)
            )
            self.se_fc2_choices_params = nn.Parameter(
                1e-3 * torch.randn(self.n_choices)
            )

            self.register_buffer("se_fc1_index", torch.FloatTensor([8]))
            self.register_buffer("se_fc1_bits_weights", torch.FloatTensor([8.0]))
            self.register_buffer("se_fc1_bits_activations", torch.FloatTensor([8.0]))

            self.register_buffer("se_fc2_index", torch.FloatTensor([8]))
            self.register_buffer("se_fc2_bits_weights", torch.FloatTensor([8.0]))
            self.register_buffer("se_fc2_bits_activations", torch.FloatTensor([8.0]))

            self.se_fc1_choices_weight = None
            self.se_fc2_choices_weight = None

            self.se_fc1_bops = []
            self.se_fc2_bops = []

        self.register_buffer("conv2_index", torch.FloatTensor([8]))
        self.register_buffer("conv2_bits_weights", torch.FloatTensor([8.0]))
        self.register_buffer("conv2_bits_activations", torch.FloatTensor([8.0]))

        self.register_buffer("conv3_index", torch.FloatTensor([8]))
        self.register_buffer("conv3_bits_weights", torch.FloatTensor([8.0]))
        self.register_buffer("conv3_bits_activations", torch.FloatTensor([8.0]))

        self.conv2_choices_weight = None
        self.conv3_choices_weight = None

        self.conv2_bops = []
        self.conv3_bops = []
        self.init_state = False

    def compute_gate_output(self, choices_params, tau):
        output = F.gumbel_softmax(choices_params, tau)
        return output

    def pre_compute_bops(self):
        bits = [2, 4, 8]
        for i in bits:
            for j in bits:
                if self.expand:
                    self.conv1_bops.append(
                        compute_bops(
                            self.conv1[-1].kernel_size[0],
                            self.conv1[-1].in_channels,
                            self.conv1[-1].out_channels // self.conv1[-1].groups,
                            self.conv1[-1].h,
                            self.conv1[-1].w,
                            i,
                            j,
                        )
                    )
                self.conv2_bops.append(
                    compute_bops(
                        self.conv2[-1].kernel_size[0],
                        self.conv2[-1].in_channels,
                        self.conv2[-1].out_channels // self.conv2[-1].groups,
                        self.conv2[-1].h,
                        self.conv2[-1].w,
                        i,
                        j,
                    )
                )
                self.conv3_bops.append(
                    compute_bops(
                        self.conv3[-1].kernel_size[0],
                        self.conv3[-1].in_channels,
                        self.conv3[-1].out_channels // self.conv3[-1].groups,
                        self.conv3[-1].h,
                        self.conv3[-1].w,
                        i,
                        j,
                    )
                )
                if self.SE:
                    self.se_fc1_bops.append(
                        compute_bops(
                            1,
                            self.squeeze_block.dense[0][-1].in_features,
                            self.squeeze_block.dense[0][-1].out_features,
                            1,
                            1,
                            i,
                            j,
                        )
                    )
                    self.se_fc2_bops.append(
                        compute_bops(
                            1,
                            self.squeeze_block.dense[2][-1].in_features,
                            self.squeeze_block.dense[2][-1].out_features,
                            1,
                            1,
                            i,
                            j,
                        )
                    )

    def forward_se(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)

        fc1_out = 0
        for i in range(self.n_choices):
            fc1_out += self.se_fc1_choices_weight[i] * self.squeeze_block.dense[0][i](
                out
            )
        out = self.squeeze_block.dense[1](fc1_out)

        fc2_out = 0
        for i in range(self.n_choices):
            fc2_out += self.se_fc2_choices_weight[i] * self.squeeze_block.dense[2][i](
                out
            )

        out = self.squeeze_block.dense[3](fc2_out)
        out = out.view(batch, channels, 1, 1)
        return x * out

    def forward(self, x):
        # MobileNetV2
        x, tau = x
        if not self.init_state:
            out = x
            if self.expand:
                out = self.conv1[-1](out)
                out = self.bn1(out)
                out = self.relu1(out)

            out = self.conv2[-1](out)
            out = self.bn2(out)

            # Squeeze and Excite
            if self.SE:
                out = self.squeeze_block(out)
            else:
                out = self.relu2(out)

            out = self.conv3[-1](out)
            out = self.bn3(out)

            self.init_state = True
            self.pre_compute_bops()

            # connection
            if self.use_connect:
                return [x + out, tau]
            else:
                return [out, tau]
        else:
            out = x
            self.conv2_choices_weight = self.compute_gate_output(
                self.conv2_choices_params, tau
            )
            self.conv3_choices_weight = self.compute_gate_output(
                self.conv3_choices_params, tau
            )
            if self.expand:
                self.conv1_choices_weight = self.compute_gate_output(
                    self.conv1_choices_params, tau
                )
                conv1_x = 0
                for i in range(self.n_choices):
                    conv1_x += self.conv1_choices_weight[i] * self.conv1[i](out)
                # out = self.conv1(out)
                out = self.bn1(conv1_x)
                out = self.relu1(out)

            conv2_x = 0
            for i in range(self.n_choices):
                conv2_x += self.conv2_choices_weight[i] * self.conv2[i](out)
            out = self.bn2(conv2_x)

            # Squeeze and Excite
            if self.SE:
                self.se_fc1_choices_weight = self.compute_gate_output(
                    self.se_fc1_choices_params, tau
                )
                self.se_fc2_choices_weight = self.compute_gate_output(
                    self.se_fc2_choices_params, tau
                )
                out = self.forward_se(out)
            else:
                out = self.relu2(out)

            conv3_x = 0
            for i in range(self.n_choices):
                conv3_x += self.conv3_choices_weight[i] * self.conv3[i](out)
            out = self.bn3(conv3_x)

            # connection
            if self.use_connect:
                return [x + out, tau]
            else:
                return [out, tau]

    def compute_bops(self):
        conv1_bops = 0
        if self.expand:
            # bops of conv1
            pre_computed_conv1_bops = self.conv1[-1].weight.data.new_tensor(
                self.conv1_bops
            )
            conv1_bops = (pre_computed_conv1_bops * self.conv1_choices_weight).sum()
        # bops of conv2
        pre_computed_conv2_bops = self.conv2[-1].weight.data.new_tensor(self.conv2_bops)
        conv2_bops = (pre_computed_conv2_bops * self.conv2_choices_weight).sum()

        pre_computed_conv3_bops = self.conv3[-1].weight.data.new_tensor(self.conv3_bops)
        conv3_bops = (pre_computed_conv3_bops * self.conv3_choices_weight).sum()

        se_bops = 0
        if self.SE:
            pre_computed_se_fc1_bops = self.conv1[-1].weight.data.new_tensor(
                self.se_fc1_bops
            )
            se_fc1_bops = (pre_computed_se_fc1_bops * self.se_fc1_choices_weight).sum()

            pre_computed_se_fc2_bops = self.conv1[-1].weight.data.new_tensor(
                self.se_fc2_bops
            )
            se_fc2_bops = (pre_computed_se_fc2_bops * self.se_fc2_choices_weight).sum()

            se_bops = se_fc1_bops + se_fc2_bops

        total_bops = conv1_bops + conv2_bops + conv3_bops + se_bops
        return total_bops

    def get_current_bops(self):
        conv1_bops = 0
        bits = [2, 4, 8]
        if self.expand:
            max_idnex = torch.argmax(self.conv1_choices_params)
            bits_weights_index = max_idnex // 3
            bits_activation_index = max_idnex % 3
            bits_weights = bits[int(bits_weights_index)]
            bits_activation = bits[int(bits_activation_index)]
            conv1_bops = compute_bops(
                self.conv1[-1].kernel_size[0],
                self.conv1[-1].in_channels,
                self.conv1[-1].out_channels // self.conv1[-1].groups,
                self.conv1[-1].h,
                self.conv1[-1].w,
                bits_weights,
                bits_activation,
            )
            self.conv1_index.data.fill_(max_idnex)
            self.conv1_bits_weights.data.fill_(bits_weights)
            self.conv1_bits_activations.data.fill_(bits_activation)

        max_idnex = torch.argmax(self.conv2_choices_params)
        bits_weights_index = max_idnex // 3
        bits_activation_index = max_idnex % 3
        bits_weights = bits[int(bits_weights_index)]
        bits_activation = bits[int(bits_activation_index)]
        conv2_bops = compute_bops(
            self.conv2[-1].kernel_size[0],
            self.conv2[-1].in_channels,
            self.conv2[-1].out_channels // self.conv2[-1].groups,
            self.conv2[-1].h,
            self.conv2[-1].w,
            bits_weights,
            bits_activation,
        )
        self.conv2_index.data.fill_(max_idnex)
        self.conv2_bits_weights.data.fill_(bits_weights)
        self.conv2_bits_activations.data.fill_(bits_activation)

        max_idnex = torch.argmax(self.conv3_choices_params)
        bits_weights_index = max_idnex // 3
        bits_activation_index = max_idnex % 3
        bits_weights = bits[int(bits_weights_index)]
        bits_activation = bits[int(bits_activation_index)]
        conv3_bops = compute_bops(
            self.conv3[-1].kernel_size[0],
            self.conv3[-1].in_channels,
            self.conv3[-1].out_channels // self.conv3[-1].groups,
            self.conv3[-1].h,
            self.conv3[-1].w,
            bits_weights,
            bits_activation,
        )
        self.conv3_index.data.fill_(max_idnex)
        self.conv3_bits_weights.data.fill_(bits_weights)
        self.conv3_bits_activations.data.fill_(bits_activation)

        se_bops = 0
        if self.SE:
            max_idnex = torch.argmax(self.se_fc1_choices_weight)
            bits_weights_index = max_idnex // 3
            bits_activation_index = max_idnex % 3
            bits_weights = bits[int(bits_weights_index)]
            bits_activation = bits[int(bits_activation_index)]
            se_fc1_bops = compute_bops(
                1,
                self.squeeze_block.dense[0][-1].in_features,
                self.squeeze_block.dense[0][-1].out_features,
                1,
                1,
                bits_weights,
                bits_activation,
            )
            self.se_fc1_index.data.fill_(max_idnex)
            self.se_fc1_bits_weights.data.fill_(bits_weights)
            self.se_fc1_bits_activations.data.fill_(bits_activation)

            max_idnex = torch.argmax(self.se_fc2_choices_weight)
            bits_weights_index = max_idnex // 3
            bits_activation_index = max_idnex % 3
            bits_weights = bits[int(bits_weights_index)]
            bits_activation = bits[int(bits_activation_index)]
            se_fc2_bops = compute_bops(
                1,
                self.squeeze_block.dense[2][-1].in_features,
                self.squeeze_block.dense[2][-1].out_features,
                1,
                1,
                bits_weights,
                bits_activation,
            )
            # print("BitsW: {}, BitsA: {}".format(bits_weights, bits_activation))
            self.se_fc2_index.data.fill_(max_idnex)
            self.se_fc2_bits_weights.data.fill_(bits_weights)
            self.se_fc2_bits_activations.data.fill_(bits_activation)

            se_bops = se_fc1_bops + se_fc2_bops
            # print("Layer: {}, BOPs: {}".format("se_fc1", se_fc1_bops))
            # print("Layer: {}, BOPs: {}".format("se_fc2", se_fc2_bops))

        # print("Layer: {}, BOPs: {}".format("conv1", conv1_bops))
        # print("Layer: {}, BOPs: {}".format("conv2", conv2_bops))
        # print("Layer: {}, BOPs: {}".format("conv3", conv3_bops))
        
        total_bops = conv1_bops + conv2_bops + conv3_bops + se_bops
        return total_bops

    def get_total_bops(self):
        conv1_bops = 0
        if self.expand:
            conv1_bops = compute_bops(
                self.conv1[-1].kernel_size[0],
                self.conv1[-1].in_channels,
                self.conv1[-1].out_channels // self.conv1[-1].groups,
                self.conv1[-1].h,
                self.conv1[-1].w,
            )
        conv2_bops = compute_bops(
            self.conv2[-1].kernel_size[0],
            self.conv2[-1].in_channels,
            self.conv2[-1].out_channels // self.conv2[-1].groups,
            self.conv2[-1].h,
            self.conv2[-1].w,
        )
        conv3_bops = compute_bops(
            self.conv3[-1].kernel_size[0],
            self.conv3[-1].in_channels,
            self.conv3[-1].out_channels // self.conv3[-1].groups,
            self.conv3[-1].h,
            self.conv3[-1].w,
        )
        se_bops = 0
        if self.SE:
            se_fc1_bops = compute_bops(
                1,
                self.squeeze_block.dense[0][-1].in_features,
                self.squeeze_block.dense[0][-1].out_features,
                1,
                1,
            )
            se_fc2_bops = compute_bops(
                1,
                self.squeeze_block.dense[2][-1].in_features,
                self.squeeze_block.dense[2][-1].out_features,
                1,
                1,
            )
            se_bops = se_fc1_bops + se_fc2_bops

        total_bops = conv1_bops + conv2_bops + conv3_bops + se_bops
        return total_bops


class DNASAsyMobileNetV3Cifar(nn.Module):
    def __init__(
        self,
        model_mode="large",
        num_classes=1000,
        multiplier=1.0,
        dropout_rate=0.2,
        quantize_first_last=False,
    ):
        super(DNASAsyMobileNetV3Cifar, self).__init__()
        self.num_classes = num_classes

        if model_mode == "large":
            layers = [
                # kernel_size, exp_size, out_channels_num, use_SE, NL, stride
                [3, 16, 16, False, "RE", 1],
                [3, 64, 24, False, "RE", 1],
                [3, 72, 24, False, "RE", 1],
                [5, 72, 40, True, "RE", 2],
                [5, 120, 40, True, "RE", 1],
                [5, 120, 40, True, "RE", 1],
                [3, 240, 80, False, "HS", 2],
                [3, 200, 80, False, "HS", 1],
                [3, 184, 80, False, "HS", 1],
                [3, 184, 80, False, "HS", 1],
                [3, 480, 112, True, "HS", 1],
                [3, 672, 112, True, "HS", 1],
                [5, 672, 160, True, "HS", 2],
                [5, 960, 160, True, "HS", 1],
                [5, 960, 160, True, "HS", 1],
            ]
        elif model_mode == "small":
            configs = [
                # kernel_size, exp_size, out_channels_num, use_SE, NL, stride
                [3, 16, 16, True, "RE", 1],
                [3, 72, 24, False, "RE", 2],
                [3, 88, 24, False, "RE", 1],
                [5, 96, 40, True, "HS", 2],
                [5, 240, 40, True, "HS", 1],
                [5, 240, 40, True, "HS", 1],
                [5, 120, 48, True, "HS", 1],
                [5, 144, 48, True, "HS", 1],
                [5, 288, 96, True, "HS", 2],
                [5, 576, 96, True, "HS", 1],
                [5, 576, 96, True, "HS", 1],
            ]

        init_conv_out = _make_divisible(16 * multiplier)
        if quantize_first_last:
            self.conv1 = QConv2d(
                in_channels=3,
                out_channels=init_conv_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                bits_weights=8,
                bits_activations=32,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=init_conv_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        self.bn1 = nn.BatchNorm2d(init_conv_out)
        self.relu1 = h_swish(inplace=True)

        self.block = []
        for kernel_size, exp_size, out_channels_num, use_SE, NL, stride in layers:
            output_channels_num = _make_divisible(out_channels_num * multiplier)
            exp_size = _make_divisible(exp_size * multiplier)
            self.block.append(
                DNASAsyMobileBlockCifar(
                    init_conv_out,
                    output_channels_num,
                    kernel_size,
                    stride,
                    NL,
                    use_SE,
                    exp_size,
                )
            )
            init_conv_out = out_channels_num
        self.block = nn.Sequential(*self.block)

        out_conv1_in = _make_divisible(160 * multiplier)
        out_conv1_out = _make_divisible(960 * multiplier)
        self.conv2 = nn.ModuleList(
            build_conv_candidate_ops(
                out_conv1_in, out_conv1_out, kernel_size=1, stride=1, bias=False,
            )
        )
        self.bn2 = nn.BatchNorm2d(out_conv1_out)
        self.relu2 = h_swish(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        out_conv2_in = _make_divisible(960 * multiplier)
        out_conv2_out = _make_divisible(1280 * multiplier)

        self.conv3 = nn.ModuleList(
            build_conv_candidate_ops(
                out_conv2_in, out_conv2_out, kernel_size=1, stride=1,
            )
        )
        self.relu3 = h_swish(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        if quantize_first_last:
            self.fc = QLinear(
                out_conv2_out, self.num_classes, bits_weights=8, bits_activations=8
            )
        else:
            self.fc = nn.Linear(out_conv2_out, self.num_classes)

        self.n_choices = 9
        self.conv2_choices_params = nn.Parameter(1e-3 * torch.randn(self.n_choices))
        self.conv3_choices_params = nn.Parameter(1e-3 * torch.randn(self.n_choices))

        self.register_buffer("conv2_index", torch.FloatTensor([0]))
        self.register_buffer("conv2_bits_weights", torch.FloatTensor([2.0]))
        self.register_buffer("conv2_bits_activations", torch.FloatTensor([2.0]))

        self.register_buffer("conv3_index", torch.FloatTensor([0]))
        self.register_buffer("conv3_bits_weights", torch.FloatTensor([2.0]))
        self.register_buffer("conv3_bits_activations", torch.FloatTensor([2.0]))

        self.conv2_choices_weight = None
        self.conv3_choices_weight = None

        self.conv2_bops = []
        self.conv3_bops = []
        self.init_state = False

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def pre_compute_bops(self):
        bits = [2, 4, 8]
        for i in bits:
            for j in bits:
                self.conv2_bops.append(
                    compute_bops(
                        self.conv2[-1].kernel_size[0],
                        self.conv2[-1].in_channels,
                        self.conv2[-1].out_channels // self.conv2[-1].groups,
                        self.conv2[-1].h,
                        self.conv2[-1].w,
                        i,
                        j,
                    )
                )
                self.conv3_bops.append(
                    compute_bops(
                        self.conv3[-1].kernel_size[0],
                        self.conv3[-1].in_channels,
                        self.conv3[-1].out_channels // self.conv3[-1].groups,
                        self.conv3[-1].h,
                        self.conv3[-1].w,
                        i,
                        j,
                    )
                )

    def compute_conv2_conv3_bops(self):
        # bops of conv2
        pre_computed_conv2_bops = self.conv2[-1].weight.data.new_tensor(self.conv2_bops)
        conv2_bops = (pre_computed_conv2_bops * self.conv2_choices_weight).sum()

        pre_computed_conv3_bops = self.conv3[-1].weight.data.new_tensor(self.conv3_bops)
        conv3_bops = (pre_computed_conv3_bops * self.conv3_choices_weight).sum()
        total_bops = conv2_bops + conv3_bops
        return total_bops

    def get_current_conv2_conv3_bops(self):
        bits = [2, 4, 8]
        max_idnex = torch.argmax(self.conv2_choices_params)
        bits_weights_index = max_idnex // 3
        bits_activation_index = max_idnex % 3
        bits_weights = bits[int(bits_weights_index)]
        bits_activation = bits[int(bits_activation_index)]
        conv2_bops = compute_bops(
            self.conv2[-1].kernel_size[0],
            self.conv2[-1].in_channels,
            self.conv2[-1].out_channels // self.conv2[-1].groups,
            self.conv2[-1].h,
            self.conv2[-1].w,
            bits_weights,
            bits_activation,
        )
        # print("Layer: {}, BOPs: {}".format("conv2", conv2_bops))
        self.conv2_index.data.fill_(max_idnex)
        self.conv2_bits_weights.data.fill_(bits_weights)
        self.conv2_bits_activations.data.fill_(bits_activation)

        max_idnex = torch.argmax(self.conv3_choices_params)
        bits_weights_index = max_idnex // 3
        bits_activation_index = max_idnex % 3
        bits_weights = bits[int(bits_weights_index)]
        bits_activation = bits[int(bits_activation_index)]
        conv3_bops = compute_bops(
            self.conv3[-1].kernel_size[0],
            self.conv3[-1].in_channels,
            self.conv3[-1].out_channels // self.conv3[-1].groups,
            self.conv3[-1].h,
            self.conv3[-1].w,
            bits_weights,
            bits_activation,
        )
        self.conv3_index.data.fill_(max_idnex)
        self.conv3_bits_weights.data.fill_(bits_weights)
        self.conv3_bits_activations.data.fill_(bits_activation)
        # print("Layer: {}, BOPs: {}".format("conv3", conv3_bops))

        total_bops = conv2_bops + conv3_bops
        return total_bops

    def get_total_conv2_conv3_bops(self):
        conv2_bops = compute_bops(
            self.conv2[-1].kernel_size[0],
            self.conv2[-1].in_channels,
            self.conv2[-1].out_channels // self.conv2[-1].groups,
            self.conv2[-1].h,
            self.conv2[-1].w,
        )
        conv3_bops = compute_bops(
            self.conv3[-1].kernel_size[0],
            self.conv3[-1].in_channels,
            self.conv3[-1].out_channels // self.conv3[-1].groups,
            self.conv3[-1].h,
            self.conv3[-1].w,
        )

        total_bops = conv2_bops + conv3_bops
        return total_bops

    def compute_gate_output(self, choices_params, tau):
        output = F.gumbel_softmax(choices_params, tau)
        return output

    def forward(self, x, tau):
        if not self.init_state:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)

            out, _ = self.block([out, tau])

            out = self.conv2[-1](out)
            out = self.bn2(out)
            out = self.relu2(out)

            out = self.avgpool(out)
            out = self.conv3[-1](out)
            out = self.relu3(out)

            out = out.view(out.size(0), -1)
            out = self.dropout(out)
            out = self.fc(out)

            self.init_state = True
            self.pre_compute_bops()
            return out
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)

            out, _ = self.block([out, tau])

            self.conv2_choices_weight = self.compute_gate_output(
                self.conv2_choices_params, tau
            )
            self.conv3_choices_weight = self.compute_gate_output(
                self.conv3_choices_params, tau
            )

            conv2_x = 0
            for i in range(self.n_choices):
                conv2_x += self.conv2_choices_weight[i] * self.conv2[i](out)
            out = self.bn2(conv2_x)
            out = self.relu2(out)

            out = self.avgpool(out)
            conv3_x = 0
            for i in range(self.n_choices):
                conv3_x += self.conv3_choices_weight[i] * self.conv3[i](out)
            out = self.relu3(conv3_x)

            out = out.view(out.size(0), -1)
            out = self.dropout(out)
            out = self.fc(out)
            return out


# temp = torch.zeros((1, 3, 224, 224))
# model = DNASAsyMobileNetV3Cifar(model_mode="LARGE", num_classes=1000, multiplier=1.0)
# print(model(temp).shape)
# print(get_model_parameters(model))
