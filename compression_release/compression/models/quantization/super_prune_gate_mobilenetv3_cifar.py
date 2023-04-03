'''
MobileNetV3 From <Searching for MobileNetV3>, arXiv:1905.02244.
Ref: https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
     https://github.com/kuan-wang/pytorch-mobilenet-v3/blob/master/mobilenetv3.py
     https://github.com/ShowLo/MobileNetV3
     
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class SuperPruneAsyMobileBlockCifar(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, nonLinear, SE, exp_size, group_size=16):
        super(SuperPruneAsyMobileBlockCifar, self).__init__()
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
            self.conv1 = nn.Conv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(exp_size)
            self.relu1 = activation(inplace=True)

        self.conv2 = nn.Conv2d(exp_size, exp_size, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp_size, bias=False)
        self.bn2 = nn.BatchNorm2d(exp_size)
        self.relu2 = activation(inplace=True)

        self.conv3 = nn.Conv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)

        self.use_connect = stride == 1 and in_channels == out_channels

        self.group_size = group_size
        self.num_choices = exp_size // self.group_size
        # self.num_choices = num_choices
        # self.group_size = intermedia_planes // self.num_choices
        self.channel_thresholds = nn.Parameter(torch.zeros(1))
        self.register_buffer("assigned_indicator", torch.zeros(self.num_choices))
        self.indicator = None
        self.init_state = False
        self.output_h = []
        self.output_w = []

    def get_bops(self):
        num_group = self.indicator.sum()
        num_channels = num_group * self.group_size
        conv1_bops = 0
        if self.expand:
            conv1_bops = compute_bops(self.conv1.kernel_size[0], self.conv1.in_channels, num_channels, self.conv1.h, self.conv1.w)
            # print("Layer: {}, BOPs: {}".format("conv1", conv1_bops))
        conv2_bops = compute_bops(self.conv2.kernel_size[0], num_channels, num_channels // num_channels, self.conv2.h, self.conv2.w)
        # print("Layer: {}, BOPs: {}".format("conv2", conv2_bops))
        conv3_bops = compute_bops(self.conv3.kernel_size[0], num_channels, self.conv3.out_channels, self.conv3.h, self.conv3.w)
        # print("Layer: {}, BOPs: {}".format("conv3", conv3_bops))
        se_fc1_bops = 0
        se_fc2_bops = 0
        if self.SE:
            se_fc1_bops = compute_bops(1, num_channels, self.squeeze_block.dense[0].out_features, 1, 1)
            # print("Layer: {}, BOPs: {}".format("se_fc1", se_fc1_bops))
            se_fc2_bops = compute_bops(1, self.squeeze_block.dense[2].in_features, num_channels, 1, 1)
            # print("Layer: {}, BOPs: {}".format("se_fc2", se_fc2_bops))
        total_bops = conv1_bops + conv2_bops + conv3_bops + se_fc1_bops + se_fc2_bops
        return total_bops

    def get_total_bops(self):
        conv1_bops = 0
        if self.expand:
            conv1_bops = compute_bops(self.conv1.kernel_size[0], self.conv1.in_channels, self.conv1.out_channels, self.conv1.h, self.conv1.w)
        conv2_bops = compute_bops(self.conv2.kernel_size[0], self.conv2.in_channels, self.conv2.out_channels // self.conv2.groups, self.conv2.h, self.conv2.w)
        conv3_bops = compute_bops(self.conv3.kernel_size[0], self.conv3.in_channels, self.conv3.out_channels, self.conv3.h, self.conv3.h)
        se_fc1_bops = 0
        se_fc2_bops = 0
        if self.SE:
            se_fc1_bops = compute_bops(1, self.squeeze_block.dense[0].in_features, self.squeeze_block.dense[0].out_features, 1, 1)
            se_fc2_bops = compute_bops(1, self.squeeze_block.dense[2].in_features, self.squeeze_block.dense[2].out_features, 1, 1)
        total_bops = conv1_bops + conv2_bops + conv3_bops + se_fc1_bops + se_fc2_bops
        return total_bops

    def compute_indicator(self):
        # TODO: check whether to compute gradient with filter
        # TODO: check l1-norm and l2-norm
        # TODO: whether to use gradient scale, need to check on ImageNet
        # TODO: add maximum pruning bound
        # TODO: whether to use quantized filter?
        if hasattr(self, "conv1"):
            filter_weight = self.conv1.weight
        else:
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
            out = x
            if self.expand:
                out = self.conv1(out)
                out = self.bn1(out)
                out = self.relu1(out)

                _, _, conv1_out_shape_h, conv1_out_shape_w = out.shape

            out = self.conv2(out)
            out = self.bn2(out)

            _, _, conv2_out_shape_h, conv2_out_shape_w = out.shape

            # Squeeze and Excite
            if self.SE:
                out = self.squeeze_block(out)
            else:
                out = self.relu2(out)

            out = self.conv3(out)
            out = self.bn3(out)

            _, _, conv3_out_shape_h, conv3_out_shape_w = out.shape

            if self.expand:
                self.output_h.append(conv1_out_shape_h)
                self.output_w.append(conv1_out_shape_w)
            self.output_h.append(conv2_out_shape_h)
            self.output_h.append(conv3_out_shape_h)
            self.output_w.append(conv2_out_shape_w)
            self.output_w.append(conv3_out_shape_w)
            self.init_state = True

            # connection
            if self.use_connect:
                return x + out
            else:
                return out
        else:
            out = x

            indicator = self.compute_indicator()
            if self.expand:
                out = self.conv1(out)
                out = self.bn1(out)
                out = self.relu1(out)

            n, c, h, w = out.shape
            reshape_out = out.reshape(n, c // self.group_size, self.group_size, h, w)
            reshape_out = reshape_out * indicator.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out = reshape_out.reshape(n, c, h, w)

            out = self.conv2(out)
            out = self.bn2(out)

            n, c, h, w = out.shape
            reshape_out = out.reshape(n, c // self.group_size, self.group_size, h, w)
            reshape_out = reshape_out * indicator.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out = reshape_out.reshape(n, c, h, w)

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


class SuperPruneAsyMobileNetV3Cifar(nn.Module):
    def __init__(self, model_mode="large", num_classes=1000, multiplier=1.0, dropout_rate=0.2, group_size=16):
        super(SuperPruneAsyMobileNetV3Cifar, self).__init__()
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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=init_conv_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(init_conv_out)
        self.relu1 = h_swish(inplace=True)

        self.block = []
        for kernel_size, exp_size, out_channels_num, use_SE, NL, stride in layers:
            output_channels_num = _make_divisible(out_channels_num * multiplier)
            exp_size = _make_divisible(exp_size * multiplier)
            self.block.append(SuperPruneAsyMobileBlockCifar(init_conv_out, output_channels_num, kernel_size, stride, NL, use_SE, exp_size, group_size=group_size))
            init_conv_out = out_channels_num
        self.block = nn.Sequential(*self.block)

        out_conv1_in = _make_divisible(160 * multiplier)
        out_conv1_out = _make_divisible(960 * multiplier)
        self.conv2 = nn.Conv2d(out_conv1_in, out_conv1_out, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_conv1_out)
        self.relu2 = h_swish(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        out_conv2_in = _make_divisible(960 * multiplier)
        out_conv2_out = _make_divisible(1280 * multiplier)

        self.conv3 = nn.Conv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1)
        self.relu3 = h_swish(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
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

    def get_total_bops(self):
        total_bops = 0
        layer = self.conv1
        bops = compute_bops(layer.kernel_size[0], layer.in_channels, layer.out_channels, layer.h, layer.w)
        total_bops += bops

        layer = getattr(self, "block")
        for name, module in layer.named_modules():
            if isinstance(module, SuperPruneAsyMobileBlockCifar):
                bops = module.get_total_bops()
                total_bops += bops

        layer = self.conv2
        bops = compute_bops(layer.kernel_size[0], layer.in_channels, layer.out_channels, layer.h, layer.w)
        total_bops += bops

        layer = self.conv3
        bops = compute_bops(layer.kernel_size[0], layer.in_channels, layer.out_channels, layer.h, layer.w)
        total_bops += bops
                    
        layer = self.fc
        bops = compute_bops(1, layer.in_features, layer.out_features, 1, 1)
        total_bops += bops
        return total_bops

    def get_bops(self):
        current_bops = 0
        layer = self.conv1
        bops = compute_bops(layer.kernel_size[0], layer.in_channels, layer.out_channels, layer.h, layer.w)
        current_bops += bops
        # print("Outer Layer: {}, BOPs: {}".format("conv1", bops))

        layer = getattr(self, "block")
        for name, module in layer.named_modules():
            if isinstance(module, SuperPruneAsyMobileBlockCifar):
                # print("Layer: {}".format(name))
                bops = module.get_bops()
                current_bops += bops

        layer = self.conv2
        bops = compute_bops(layer.kernel_size[0], layer.in_channels, layer.out_channels, layer.h, layer.w)
        current_bops += bops
        # print("Outer Layer: {}, BOPs: {}".format("conv2", bops))

        layer = self.conv3
        bops = compute_bops(layer.kernel_size[0], layer.in_channels, layer.out_channels, layer.h, layer.w)
        current_bops += bops
        # print("Outer Layer: {}, BOPs: {}".format("conv3", bops))

        layer = self.fc
        bops = compute_bops(1, layer.in_features, layer.out_features, 1, 1)
        current_bops += bops
        # print("Outer Layer: {}, BOPs: {}".format("fc", bops))
        return current_bops

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
# model = SuperPruneAsyMobileNetV3Cifar(model_mode="LARGE", num_classes=1000, multiplier=1.0)
# print(model(temp).shape)
# print(get_model_parameters(model))
