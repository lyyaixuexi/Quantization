"""MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""
import torch.nn as nn
import torch.nn.functional as F
from compression.models.quantization.dorefa import QConv2d, QLinear
from compression.models.quantization.dq_conv import DQConv2d

__all__ = [
    "DQMobileNetV2CifarBlock",
    "DQMobileNetV2Cifar",
    "dqmobilenetv2_cifar",
]


class DQMobileNetV2CifarBlock(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(
        self,
        in_planes,
        out_planes,
        expansion,
        stride,
    ):
        super(DQMobileNetV2CifarBlock, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = DQConv2d(
            in_planes,
            planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DQConv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=planes,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = DQConv2d(
            planes,
            out_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                DQConv2d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class DQMobileNetV2Cifar(nn.Module):
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
    ):
        super(DQMobileNetV2Cifar, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10

        if quantize_first_last:
            self.conv1 = QConv2d(
                3,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                bits_weights=8, 
                bits_activations=32
            )
        else:
            self.conv1 = nn.Conv2d(
                3, 32, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(
            in_planes=32,
        )
        self.conv2 = DQConv2d(
            320,
            1280,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(1280)
        if quantize_first_last:
            self.linear = QLinear(1280, num_classes, bits_weights=8, bits_activations=8)
        else:
            self.linear = nn.Linear(1280, num_classes)

    def _make_layers(
        self, in_planes, 
    ):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(
                    DQMobileNetV2CifarBlock(
                        in_planes,
                        out_planes,
                        expansion,
                        stride,
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


def dqmobilenetv2_cifar(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DQMobileNetV2Cifar(**kwargs)
    return model

