# copy from pytorch-torchvision-models-resnet
import torch.nn as nn
from compression.models.quantization.super_quan_conv_latency import SuperQLatencyConv2d
from compression.models.quantization.dorefa_clip import QConv2d, QLinear

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "BasicBlock",
    "Bottleneck",
]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def qconv3x3(in_planes, out_planes, stride=1, latency_dict=None, power_dict=None):
    "3x3 convolution with padding"
    return SuperQLatencyConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, latency_dict=latency_dict, power_dict=power_dict)


class SuperQuanLatencyBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, latency_dict=None, power_dict=None):
        super(SuperQuanLatencyBasicBlock, self).__init__()
        self.name = "resnet-basic"
        self.conv1 = qconv3x3(inplanes, planes, stride, latency_dict=latency_dict, power_dict=power_dict)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = qconv3x3(planes, planes, latency_dict=latency_dict, power_dict=power_dict)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.block_index = 0

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SuperQuanLatencyBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, latency_dict=None, power_dict=None):
        super(SuperQuanLatencyBottleneck, self).__init__()
        self.name = "resnet-bottleneck"
        self.conv1 = SuperQLatencyConv2d(inplanes, planes, kernel_size=1, bias=False, latency_dict=latency_dict, power_dict=power_dict)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SuperQLatencyConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, latency_dict=latency_dict, power_dict=power_dict)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SuperQLatencyConv2d(planes, planes * 4, kernel_size=1, bias=False, latency_dict=latency_dict, power_dict=power_dict)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.block_index = 0

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


class SuperQuanLatencyResNet(nn.Module):
    def __init__(self, depth, num_classes=1000, latency_dict=None, power_dict=None):
        self.inplanes = 64
        super(SuperQuanLatencyResNet, self).__init__()
        if depth < 50:
            block = SuperQuanLatencyBasicBlock
        else:
            block = SuperQuanLatencyBottleneck

        if depth == 18:
            layers = [2, 2, 2, 2]
        elif depth == 34:
            layers = [3, 4, 6, 3]
        elif depth == 50:
            layers = [3, 4, 6, 3]
        elif depth == 101:
            layers = [3, 4, 23, 3]
        elif depth == 152:
            layers = [3, 8, 36, 3]

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = QConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, bits_weights=8, bits_activations=32)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], latency_dict=latency_dict, power_dict=power_dict)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, latency_dict=latency_dict, power_dict=power_dict)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, latency_dict=latency_dict, power_dict=power_dict)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, latency_dict=latency_dict, power_dict=power_dict)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = QLinear(512 * block.expansion, num_classes, bits_weights=8, bits_activations=8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, latency_dict=None, power_dict=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SuperQLatencyConv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    latency_dict=latency_dict,
                    power_dict=power_dict
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, latency_dict=latency_dict, power_dict=power_dict))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, latency_dict=latency_dict, power_dict=power_dict))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
