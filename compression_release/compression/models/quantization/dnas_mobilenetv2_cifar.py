"""MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""
import torch.nn as nn
import torch.nn.functional as F
from compression.models.quantization.dorefa import QConv2d, QLinear
from compression.utils.utils import *

__all__ = [
    "DNASMobileNetV2CifarBlock",
    "DNASMobileNetV2Cifar",
    "qmobilenetv2_cifar",
]


def build_conv_candidate_ops(in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, bias=True):
    candidate = []
    bits = [2, 3, 4, 5]
    for i in bits:
        for j in bits:
            candidate.append(QConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, bits_weights=i, bits_activations=j))
    return candidate


class DNASMobileNetV2CifarBlock(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(
        self,
        in_planes,
        out_planes,
        expansion,
        stride,
    ):
        super(DNASMobileNetV2CifarBlock, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        # self.conv1 = conv_type(
        #     in_planes,
        #     planes,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     bias=False,
        #     bits_weights=bits_weights,
        #     bits_activations=bits_activations,
        # )
        self.conv1 = nn.ModuleList(
            build_conv_candidate_ops(
                in_channels=in_planes,
                out_channels=planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        )
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = conv_type(
        #     planes,
        #     planes,
        #     kernel_size=3,
        #     stride=stride,
        #     padding=1,
        #     groups=planes,
        #     bias=False,
        #     bits_weights=bits_weights,
        #     bits_activations=bits_activations,
        # )
        self.conv2 = nn.ModuleList(
            build_conv_candidate_ops(
                in_channels=planes,
                out_channels=planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=planes,
                bias=False,
            )
        )
        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = conv_type(
        #     planes,
        #     out_planes,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     bias=False,
        #     bits_weights=bits_weights,
        #     bits_activations=bits_activations,
        # )
        self.conv3 = nn.ModuleList(
            build_conv_candidate_ops(
                in_channels=planes,
                out_channels=out_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        )
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                # conv_type(
                #     in_planes,
                #     out_planes,
                #     kernel_size=1,
                #     stride=1,
                #     padding=0,
                #     bias=False,
                #     bits_weights=bits_weights,
                #     bits_activations=bits_activations,
                # ),
                nn.ModuleList(
                    build_conv_candidate_ops(
                        in_channels=in_planes,
                        out_channels=out_planes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    )
                ),
                nn.BatchNorm2d(out_planes),
            )

        self.n_choices = 16
        if self.expand:
            self.conv1_choices_params = nn.Parameter(1e-3 * torch.randn(self.n_choices))
            self.register_buffer("conv1_index", torch.FloatTensor([8]))
            self.register_buffer("conv1_bits_weights", torch.FloatTensor([8.0]))
            self.register_buffer("conv1_bits_activations", torch.FloatTensor([8.0]))
            self.conv1_choices_weight = None
            self.conv1_bops = []
        self.conv2_choices_params = nn.Parameter(1e-3 * torch.randn(self.n_choices))
        self.conv3_choices_params = nn.Parameter(1e-3 * torch.randn(self.n_choices))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class DNASMobileNetV2Cifar(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(
        self,
        num_classes=10,
        quantize_first_last=False,
        bits_weights=32,
        bits_activations=32,
        quan_type="LIQ",
    ):
        super(DNASMobileNetV2Cifar, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10

        conv_type, fc_type = get_conv_fc_quan_type(quan_type)
        if quantize_first_last:
            self.conv1 = conv_type(
                3,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                bits_weights=8,
                bits_activations=32,
            )
        else:
            self.conv1 = nn.Conv2d(
                3, 32, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(
            in_planes=32,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
            conv_type=conv_type,
        )
        self.conv2 = conv_type(
            320,
            1280,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            bits_weights=bits_weights,
            bits_activations=bits_activations,
        )
        self.bn2 = nn.BatchNorm2d(1280)
        if quantize_first_last:
            self.linear = fc_type(1280, num_classes, bits_weights=8, bits_activations=8)
        else:
            self.linear = nn.Linear(1280, num_classes)

    def _make_layers(
        self, in_planes, bits_weights=32, bits_activations=32, conv_type=LIQ.QConv2d,
    ):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(
                    DNASMobileNetV2CifarBlock(
                        in_planes,
                        out_planes,
                        expansion,
                        stride,
                        bits_weights=bits_weights,
                        bits_activations=bits_activations,
                        conv_type=conv_type,
                    )
                )
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def qmobilenetv2_cifar(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DNASMobileNetV2Cifar(**kwargs)
    return model

