import math

import numpy as np
import torch.nn as nn
from compression.models.quantization.super_quan_conv import SuperQConv2d
from compression.models.quantization.dorefa_clip import QConv2d, QLinear

__all__ = ["SuperQMobileNetV1"]


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


def superqconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return SuperQConv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)


def qconv1x1(in_planes, out_planes, bits_weights=32, bits_activations=32):
    return QConv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        bias=False,
        bits_weights=bits_weights,
        bits_activations=bits_activations,
    )


def superqconv1x1(in_planes, out_planes):
    "1x1 convolution"
    return SuperQConv2d(in_planes, out_planes, kernel_size=1, bias=False)


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


def superqdwconv3x3(in_planes, out_planes, stride=1):
    "3x3 depth wise convolution"
    return SuperQConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=in_planes,
        bias=False,
    )


class SuperQMobileNetV1(nn.Module):
    """
    MobileNetV1 on ImageNet
    """

    def __init__(
        self, num_classes=1000, wide_scale=1.0, quantize_first_last=False, num_choices=8
    ):
        super(SuperQMobileNetV1, self).__init__()

        # define network structure

        self.layer_width = np.array([32, 64, 128, 256, 512, 1024])
        self.layer_width = np.around(self.layer_width * wide_scale)
        self.layer_width = self.layer_width.astype(int)
        self.segment_layer = [1, 1, 2, 2, 6, 1]  # number of layers in each segment
        self.down_sample = [
            1,
            2,
            3,
            4,
        ]  # the place of down_sample, related to segment_layer

        self.num_choices = num_choices
        self.group_size = []
        self.features = self._make_layer(quantize_first_last=quantize_first_last)
        self.dropout = nn.Dropout(0)
        if quantize_first_last:
            self.classifier = QLinear(self.layer_width[-1].item(), num_classes, bits_weights=8, bits_activations=8)
        else:
            self.classifier = nn.Linear(self.layer_width[-1].item(), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, quantize_first_last=False):
        if quantize_first_last:
            layer_list = [
                qconv3x3(in_planes=3, out_planes=self.layer_width[0].item(), stride=2, bits_weights=8, bits_activations=32),
                nn.BatchNorm2d(self.layer_width[0].item()),
                nn.ReLU(inplace=True),
            ]
        else:
            layer_list = [
                conv3x3(in_planes=3, out_planes=self.layer_width[0].item(), stride=2),
                nn.BatchNorm2d(self.layer_width[0].item()),
                nn.ReLU(inplace=True),
            ]
        for i, layer_num in enumerate(self.segment_layer):
            in_planes = self.layer_width[i].item()
            for j in range(layer_num):
                if j == layer_num - 1 and i < len(self.layer_width) - 1:
                    out_planes = self.layer_width[i + 1].item()
                else:
                    out_planes = in_planes
                if i in self.down_sample and j == layer_num - 1:
                    stride = 2
                else:
                    stride = 1
                layer_list.append(
                    superqdwconv3x3(
                        in_planes,
                        in_planes,
                        stride=stride,
                    )
                )
                layer_list.append(nn.BatchNorm2d(in_planes))
                layer_list.append(nn.ReLU(inplace=True))
                layer_list.append(
                    superqconv1x1(
                        in_planes,
                        out_planes,
                    )
                )
                layer_list.append(nn.BatchNorm2d(out_planes))
                layer_list.append(nn.ReLU(inplace=True))
                in_planes = out_planes
        layer_list.append(nn.AvgPool2d(7))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        """
        forward propagation
        """
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)

        return out


if __name__ == "__main__":
    model = SuperQMobileNetV1()
    i = 0
    for name, module in model.features.named_modules():
        if isinstance(module, nn.Conv2d):
            print("index: {}, name: {}".format(i, name))
            i += 1
    # print(model)