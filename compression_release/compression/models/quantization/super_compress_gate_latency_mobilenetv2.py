import math
import torch

import numpy as np
import torch.nn as nn
from compression.models.quantization.super_quan_conv_latency import SuperQLatencyConv2d
from compression.models.quantization.dorefa_clip import QConv2d, QLinear
from compression.utils.utils import *

__all__ = ["SuperCompressedLatencyMobileNetV2", "SuperCompressedGateLatencyMobileBottleneck"]


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


def superqconv3x3(in_planes, out_planes, stride=1, latency_dict=None, power_dict=None):
    "3x3 convolution with padding"
    return SuperQLatencyConv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
        latency_dict=latency_dict, power_dict=power_dict
    )


def conv1x1(in_planes, out_planes):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)


def qconv1x1(in_planes, out_planes, bits_weights=32, bits_activations=32):
    "1x1 convolution"
    return QConv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        bias=False,
        bits_weights=bits_weights,
        bits_activations=bits_activations,
    )


def superqconv1x1(in_planes, out_planes, latency_dict=None, power_dict=None):
    "1x1 convolution"
    return SuperQLatencyConv2d(in_planes, out_planes, kernel_size=1, bias=False,
        latency_dict=latency_dict, power_dict=power_dict)


def dwconv3x3(in_planes, out_planes, stride=1):
    "3x3 depth wise convolution"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=in_planes,
        bias=False,
    )


def qdwconv3x3(in_planes, out_planes, stride=1, bits_weights=32, bits_activations=32):
    "3x3 depth wise convolution"
    return QConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=in_planes,
        bias=False,
        bits_weights=bits_weights,
        bits_activations=bits_activations,
    )


def superqdwconv3x3(in_planes, out_planes, stride=1, latency_dict=None, power_dict=None):
    "3x3 depth wise convolution"
    return SuperQLatencyConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=in_planes,
        bias=False,
        latency_dict=latency_dict, 
        power_dict=power_dict
    )


class SuperCompressedGateLatencyMobileBottleneck(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        expand=1,
        stride=1,
        # num_choices=8
        group_size=16,
        latency_dict=None, 
        power_dict=None
    ):
        super(SuperCompressedGateLatencyMobileBottleneck, self).__init__()
        self.name = "mobile-bottleneck"
        self.expand = expand

        intermedia_planes = in_planes * expand
        if self.expand != 1:
            self.conv1 = superqconv1x1(
                in_planes,
                intermedia_planes,
                latency_dict=latency_dict, 
                power_dict=power_dict
            )
            self.bn1 = nn.BatchNorm2d(intermedia_planes)
            self.relu1 = nn.ReLU6(inplace=True)

        self.conv2 = superqdwconv3x3(
            intermedia_planes,
            intermedia_planes,
            stride=stride,
            latency_dict=latency_dict, 
            power_dict=power_dict
        )
        self.bn2 = nn.BatchNorm2d(intermedia_planes)
        self.relu2 = nn.ReLU6(inplace=True)

        self.conv3 = superqconv1x1(intermedia_planes, out_planes,
            latency_dict=latency_dict, 
            power_dict=power_dict
            )
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = stride == 1 and in_planes == out_planes
        self.block_index = 0

        self.group_size = max(intermedia_planes // 64, 4)
        self.n_choices = intermedia_planes // self.group_size
        # self.num_choices = num_choices
        # self.group_size = intermedia_planes // self.num_choices
        self.channel_thresholds = nn.Parameter(torch.zeros(1))
        self.register_buffer("assigned_indicator", torch.zeros(self.n_choices))
        self.indicator = None
        self.init_state = False
        self.output_h = []
        self.output_w = []

    def compress_fix_activation_compute_weight_latency(self):
        # num_group = self.indicator.sum()
        # channel_num = num_group * self.group_size
        indicator = self.indicator
        sort_indicator, _ = torch.topk(indicator, k=indicator.nelement())
        # latency of conv1
        conv1_latency = 0
        if self.expand != 1:
            conv1_latency = self.conv1.compress_fix_activation_compute_weight_latency(sort_indicator, group_size=self.group_size, num_choices=self.n_choices, is_out_channel=True)
        # latency of conv2
        conv2_latency = self.conv2.compress_fix_activation_compute_weight_latency(sort_indicator, group_size=self.group_size, num_choices=self.n_choices,  is_out_channel=True)
        conv3_latency = self.conv3.compress_fix_activation_compute_weight_latency(sort_indicator, group_size=self.group_size, num_choices=self.n_choices, is_out_channel=False)
        total_latency = conv1_latency + conv2_latency + conv3_latency
        return total_latency

    def compress_fix_weight_compute_activation_latency(self):
        # num_group = self.indicator.sum()
        # channel_num = num_group * self.group_size
        indicator = self.indicator
        sort_indicator, _ = torch.topk(indicator, k=indicator.nelement())
        # latency of conv1
        conv1_latency = 0
        if self.expand != 1:
            conv1_latency = self.conv1.compress_fix_weight_compute_activation_latency(sort_indicator, group_size=self.group_size, num_choices=self.n_choices, is_out_channel=True)
        # latency of conv2
        conv2_latency = self.conv2.compress_fix_weight_compute_activation_latency(sort_indicator, group_size=self.group_size, num_choices=self.n_choices,  is_out_channel=True)
        conv3_latency = self.conv3.compress_fix_weight_compute_activation_latency(sort_indicator, group_size=self.group_size, num_choices=self.n_choices,  is_out_channel=False)
        total_latency = conv1_latency + conv2_latency + conv3_latency
        return total_latency

    def compress_fix_activation_compute_weight_energy(self):
        indicator = self.indicator
        sort_indicator, _ = torch.sort(indicator, descending=True)
        # energy of conv1
        conv1_energy = 0
        if self.expand != 1:
            conv1_energy = self.conv1.compress_fix_activation_compute_weight_energy(sort_indicator, group_size=self.group_size, num_choices=self.n_choices, is_out_channel=True)
        # energy of conv2
        conv2_energy = self.conv2.compress_fix_activation_compute_weight_energy(sort_indicator, group_size=self.group_size, num_choices=self.n_choices, is_out_channel=True)
        conv3_energy = self.conv3.compress_fix_activation_compute_weight_energy(sort_indicator, group_size=self.group_size, num_choices=self.n_choices, is_out_channel=False)
        total_energy = conv1_energy + conv2_energy + conv3_energy
        return total_energy

    def compress_fix_weight_compute_activation_energy(self):
        # num_group = self.indicator.sum()
        # channel_num = num_group * self.group_size
        indicator = self.indicator
        sort_indicator, _ = torch.sort(indicator, descending=True)
        # energy of conv1
        conv1_energy = 0
        if self.expand != 1:
            conv1_energy = self.conv1.compress_fix_weight_compute_activation_energy(sort_indicator, group_size=self.group_size, num_choices=self.n_choices, is_out_channel=True)
        # energy of conv2
        conv2_energy = self.conv2.compress_fix_weight_compute_activation_energy(sort_indicator, group_size=self.group_size, num_choices=self.n_choices, is_out_channel=True)
        conv3_energy = self.conv3.compress_fix_weight_compute_activation_energy(sort_indicator, group_size=self.group_size, num_choices=self.n_choices, is_out_channel=False)
        total_energy = conv1_energy + conv2_energy + conv3_energy
        return total_energy

    def compute_indicator(self):
        # TODO: check whether to compute gradient with filter
        # TODO: check l1-norm and l2-norm
        # TODO: whether to use gradient scale, need to check on ImageNet
        # TODO: add maximum pruning bound
        # TODO: whether to use quantized filter?
        filter_weight = self.conv2.weight
        # quantized_weight = normalize_and_quantize_weight(filter_weight, self.conv1.bits_weights, self.conv1.weight_clip_value.detach())
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
        if not self.init_state:
            if self.shortcut:
                residual = x

            out = x
            if self.expand != 1:
                out = self.conv1(out)
                out = self.bn1(out)
                out = self.relu1(out)

                _, _, conv1_out_shape_h, conv1_out_shape_w = out.shape

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu2(out)

            _, _, conv2_out_shape_h, conv2_out_shape_w = out.shape

            out = self.conv3(out)
            out = self.bn3(out)

            _, _, conv3_out_shape_h, conv3_out_shape_w = out.shape

            if self.shortcut:
                out = out + residual
            if self.expand != 1:
                self.output_h.append(conv1_out_shape_h)
                self.output_w.append(conv1_out_shape_w)
            self.output_h.append(conv2_out_shape_h)
            self.output_h.append(conv3_out_shape_h)
            self.output_w.append(conv2_out_shape_w)
            self.output_w.append(conv3_out_shape_w)
            self.init_state = True
        else:
            indicator = self.compute_indicator()
            if self.shortcut:
                residual = x

            out = x
            n, _, _, _ = self.conv2.weight.shape
            indicator = indicator.reshape(-1, 1)
            indicator = indicator.expand(n // self.group_size, self.group_size).reshape(n)
            index = (indicator > 0).nonzero().squeeze()
            if self.expand != 1:
                selected_weight = torch.index_select(self.conv1.weight, 0, index)
                weight_mean = selected_weight.data.mean()
                weight_std = selected_weight.data.std()

                out = self.conv1(out, self.conv1.weight, weight_mean, weight_std)
                out = self.bn1(out)
                out = self.relu1(out)

            selected_weight = torch.index_select(self.conv2.weight, 0, index)
            weight_mean = selected_weight.data.mean()
            weight_std = selected_weight.data.std()

            out = self.conv2(out, self.conv2.weight, weight_mean, weight_std)
            out = self.bn2(out)
            out = self.relu2(out)

            selected_weight = torch.index_select(self.conv3.weight, 1, index)
            weight_mean = selected_weight.data.mean()
            weight_std = selected_weight.data.std()
            out = out * indicator.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            out = self.conv3(out, self.conv3.weight, weight_mean, weight_std)
            out = self.bn3(out)

            if self.shortcut:
                out = out + residual
        
        return out


class SuperCompressedLatencyMobileNetV2(nn.Module):
    """
    MobileNet_v2
    """

    def __init__(self, num_classes=1000, wide_scale=1.0, quantize_first_last=False, group_size=16, latency_dict=None, power_dict=None):
        super(SuperCompressedLatencyMobileNetV2, self).__init__()

        block = SuperCompressedGateLatencyMobileBottleneck
        # define network structure
        self.layer_width = np.array([32, 16, 24, 32, 64, 96, 160, 320])
        self.layer_width = np.around(self.layer_width * wide_scale)
        self.layer_width = self.layer_width.astype(int)

        self.in_planes = self.layer_width[0].item()
        if quantize_first_last:
            self.conv1 = qconv3x3(3, self.in_planes, stride=2, bits_weights=8, bits_activations=32)
        else:
            self.conv1 = conv3x3(3, self.in_planes, stride=2)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu1 = nn.ReLU6(inplace=True)
        self.layer1 = self._make_layer(block, self.layer_width[1].item(), blocks=1, group_size=group_size, latency_dict=latency_dict, power_dict=power_dict)
        self.layer2 = self._make_layer(
            block, self.layer_width[2].item(), blocks=2, expand=6, stride=2, group_size=group_size,
            latency_dict=latency_dict, power_dict=power_dict
        )
        self.layer3 = self._make_layer(
            block, self.layer_width[3].item(), blocks=3, expand=6, stride=2, group_size=group_size,
            latency_dict=latency_dict, power_dict=power_dict
        )
        self.layer4 = self._make_layer(
            block, self.layer_width[4].item(), blocks=4, expand=6, stride=2, group_size=group_size,
            latency_dict=latency_dict, power_dict=power_dict
        )
        self.layer5 = self._make_layer(
            block, self.layer_width[5].item(), blocks=3, expand=6, stride=1, group_size=group_size,
            latency_dict=latency_dict, power_dict=power_dict
        )
        self.layer6 = self._make_layer(
            block, self.layer_width[6].item(), blocks=3, expand=6, stride=2, group_size=group_size,
            latency_dict=latency_dict, power_dict=power_dict
        )
        self.layer7 = self._make_layer(
            block, self.layer_width[7].item(), blocks=1, expand=6, group_size=group_size,
            latency_dict=latency_dict, power_dict=power_dict
        )
        self.conv2 = superqconv1x1(in_planes=self.layer_width[7].item(), out_planes=1280, latency_dict=latency_dict, power_dict=power_dict)
        self.bn2 = nn.BatchNorm2d(1280)
        self.relu2 = nn.ReLU6(inplace=True)
        # self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0)

        if quantize_first_last:
            self.fc = QLinear(1280, num_classes, bits_weights=8, bits_activations=8)
        else:
            self.fc = nn.Linear(1280, num_classes)
        # self.conv3 = conv1x1(1280, num_classes)
        # self.relu3 = nn.ReLU6(inplace=True)
        # self.fc = nn.Linear(self.layer_width[8], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_planes, blocks, expand=1, stride=1, group_size=16, latency_dict=None, power_dict=None):
        layers = []
        layers.append(block(self.in_planes, out_planes, expand=expand, stride=stride, group_size=group_size, latency_dict=latency_dict, power_dict=power_dict))
        self.in_planes = out_planes
        for i in range(1, blocks):
            layers.append(block(self.in_planes, out_planes, expand=expand, group_size=group_size, latency_dict=latency_dict, power_dict=power_dict))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward propagation
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # x = self.avgpool(x)
        # # x = self.conv3(x)
        x = x.mean([2, 3])
        x = self.dropout(x)
        x = self.fc(x)

        return x
