'''
MobileNetV3 From <Searching for MobileNetV3>, arXiv:1905.02244.
Ref: https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
     https://github.com/kuan-wang/pytorch-mobilenet-v3/blob/master/mobilenetv3.py
     https://github.com/ShowLo/MobileNetV3
     
'''

import torch.nn as nn
import torch.nn.functional as F

from compression.models.quantization.dorefa_clip_wo_norm import QConv2d, QLinear


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
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., self.inplace) / 6.
        return out * x


def _make_divisible(v, divisor=8, min_value=None):
    '''
    Ensure that 'number' can be 'divisor' divisible
    Reference from original tensorflow repo:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    '''
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size),
            h_sigmoid()
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        return out * x


class QMobileBlockCifar(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, nonLinear, SE, exp_size, bits_weights=32, bits_activations=32):
        super(QMobileBlockCifar, self).__init__()
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
            self.conv1 = QConv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False, bits_weights=bits_weights, bits_activations=bits_activations)
            self.bn1 = nn.BatchNorm2d(exp_size)
            self.relu1 = activation(inplace=True)

        self.conv2 = QConv2d(exp_size, exp_size, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp_size, bias=False, bits_weights=bits_weights, bits_activations=bits_activations)
        self.bn2 = nn.BatchNorm2d(exp_size)
        self.relu2 = activation(inplace=True)

        self.conv3 = QConv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0, bias=False, bits_weights=bits_weights, bits_activations=bits_activations)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)

        self.use_connect = stride == 1 and in_channels == out_channels

    def forward(self, x):
        # MobileNetV2
        out = x
        if self.expand:
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Squeeze and Excite
        if self.SE:
            out = self.squeeze_block(out)
        else:
            out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # connection
        if self.use_connect:
            return x + out
        else:
            return out


class QMobileNetV3Cifar(nn.Module):
    def __init__(self, model_mode="large", num_classes=1000, multiplier=1.0, dropout_rate=0.2, bits_weights=32, bits_activations=32, quantize_first_last=False):
        super(QMobileNetV3Cifar, self).__init__()
        self.num_classes = num_classes

        if model_mode == "large":
            layers = [
                #kernel_size, exp_size, out_channels_num, use_SE, NL, stride
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 1],
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
                [5, 960, 160, True, 'HS', 1]
            ]
        elif model_mode == "small":
            configs = [
                #kernel_size, exp_size, out_channels_num, use_SE, NL, stride
                [3, 16, 16, True, 'RE', 1],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1]
            ]

        init_conv_out = _make_divisible(16 * multiplier)
        if quantize_first_last:
            self.conv1 = QConv2d(in_channels=3, out_channels=init_conv_out, kernel_size=3, stride=1, padding=1, bias=False, bits_weights=8, bits_activations=32)
        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=init_conv_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(init_conv_out)
        self.relu1 = h_swish(inplace=True)

        self.block = []
        for kernel_size, exp_size, out_channels_num, use_SE, NL, stride in layers:
            output_channels_num = _make_divisible(out_channels_num * multiplier)
            exp_size = _make_divisible(exp_size * multiplier)
            self.block.append(QMobileBlockCifar(init_conv_out, output_channels_num, kernel_size, stride, NL, use_SE, exp_size, bits_weights=bits_weights, bits_activations=bits_activations))
            init_conv_out = out_channels_num
        self.block = nn.Sequential(*self.block)

        out_conv1_in = _make_divisible(160 * multiplier)
        out_conv1_out = _make_divisible(960 * multiplier)
        self.conv2 = QConv2d(out_conv1_in, out_conv1_out, kernel_size=1, stride=1, bias=False, bits_weights=bits_weights, bits_activations=bits_activations)
        self.bn2 = nn.BatchNorm2d(out_conv1_out)
        self.relu2 = h_swish(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        out_conv2_in = _make_divisible(960 * multiplier)
        out_conv2_out = _make_divisible(1280 * multiplier)

        self.conv3 = QConv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1, bits_weights=bits_weights, bits_activations=bits_activations)
        self.relu3 = h_swish(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        if quantize_first_last:
            self.fc = QLinear(out_conv2_out, self.num_classes, bits_weights=8, bits_activations=8)
        else:
            self.fc = nn.Linear(out_conv2_out, self.num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        '''
        Initialize the weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.block(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.avgpool(out)
        out = self.conv3(out)
        out = self.relu3(out)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# temp = torch.zeros((1, 3, 224, 224))
# model = QMobileNetV3Cifar(model_mode="LARGE", num_classes=1000, multiplier=1.0)
# print(model(temp).shape)
# print(get_model_parameters(model))