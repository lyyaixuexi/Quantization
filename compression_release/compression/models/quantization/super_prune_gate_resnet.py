# copy from pytorch-torchvision-models-resnet
import torch
import torch.nn as nn

from compression.utils.utils import *

__all__ = [
    "SuperPrunedGateResNet",
    "SuperPrunedGateBasicBlock",
    "SuperPrunedGateBottleneck",
]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class SuperPrunedGateBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SuperPrunedGateBasicBlock, self).__init__()
        self.name = "resnet-superpruned-basic"
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.block_index = 0

        self.group_size = 4
        self.n_choices = planes // self.group_size
        self.channel_thresholds = nn.Parameter(torch.zeros(1))
        self.register_buffer("assigned_indicator", torch.zeros(self.n_choices))
        self.conv1_bops = 0
        self.conv2_bops = 0
        self.indicator = None
        self.init_state = False
        self.output_h = []
        self.output_w = []

    def get_bops(self):
        num_group = self.indicator.sum()
        num_channels = num_group * self.group_size
        conv1_bops = compute_bops(
            self.conv1.kernel_size[0],
            self.conv1.in_channels,
            num_channels,
            self.output_h[0],
            self.output_w[0],
        )
        conv2_bops = compute_bops(
            self.conv2.kernel_size[0],
            num_channels,
            self.conv2.out_channels,
            self.output_h[1],
            self.output_w[1],
        )
        downsample_bops = 0
        if self.downsample is not None:
            downsample_bops = compute_bops(
                self.downsample[0].kernel_size[0],
                self.downsample[0].in_channels,
                self.downsample[0].out_channels,
                self.downsample[0].h,
                self.downsample[0].w,
            )
        total_bops = conv1_bops + conv2_bops + downsample_bops
        return total_bops

    def get_total_bops(self):
        conv1_bops = compute_bops(
            self.conv1.kernel_size[0],
            self.conv1.in_channels,
            self.conv1.out_channels,
            self.output_h[0],
            self.output_w[0],
        )
        conv2_bops = compute_bops(
            self.conv2.kernel_size[0],
            self.conv2.in_channels,
            self.conv2.out_channels,
            self.output_h[1],
            self.output_w[1],
        )
        downsample_bops = 0
        if self.downsample is not None:
            downsample_bops = compute_bops(
                self.downsample[0].kernel_size[0],
                self.downsample[0].in_channels,
                self.downsample[0].out_channels,
                self.downsample[0].h,
                self.downsample[0].w,
            )
        total_bops = conv1_bops + conv2_bops + downsample_bops
        return total_bops

    def compute_norm(self, filter_weight):
        n, c, h, w = filter_weight.shape
        filter_weight = filter_weight.reshape(
            n // self.group_size, self.group_size * c * h * w
        )
        normalized_filter_weight = filter_weight / torch.max(torch.abs(filter_weight))
        # l2-norm
        # normalized_filter_weight_norm = (normalized_filter_weight * normalized_filter_weight).sum(1) / (self.group_size * c * h * w)
        # l1-norm
        normalized_filter_weight_norm = normalized_filter_weight.abs().sum(1) / (
            self.group_size * c * h * w
        )
        # normalized_filter_weight_norm = (normalized_filter_weight.abs().sum(1) / (self.group_size * c * h * w)).detach()
        return normalized_filter_weight_norm

    def compute_indicator(self):
        # TODO: check whether to compute gradient with filter
        # TODO: check l1-norm and l2-norm
        # TODO: whether to use gradient scale, need to check on ImageNet
        # TODO: add maximum pruning bound
        filter_weight = self.conv1.weight
        normalized_filter_weight_norm = self.compute_norm(filter_weight)
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
            residual = x

            out = self.conv1(x)
            _, _, conv1_out_shape_h, conv1_out_shape_w = out.shape
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            _, _, conv2_out_shape_h, conv2_out_shape_w = out.shape
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            self.output_h.append(conv1_out_shape_h)
            self.output_h.append(conv2_out_shape_h)
            self.output_w.append(conv1_out_shape_w)
            self.output_w.append(conv2_out_shape_w)

            out += residual
            out = self.relu(out)
            self.init_state = True
        else:
            indicator = self.compute_indicator()
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            n, c, h, w = out.shape
            reshape_out = out.reshape(n, c // self.group_size, self.group_size, h, w)
            reshape_out = reshape_out * indicator.unsqueeze(0).unsqueeze(-1).unsqueeze(
                -1
            ).unsqueeze(-1)
            out = reshape_out.reshape(n, c, h, w)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

        return out


class SuperPrunedGateBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SuperPrunedGateBottleneck, self).__init__()
        self.name = "resnet-superpruned-bottleneck"
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.block_index = 0

        self.group_size = 4
        self.n_choices = planes // self.group_size
        self.channel_thresholds_1 = nn.Parameter(torch.zeros(1))
        self.channel_thresholds_2 = nn.Parameter(torch.zeros(1))
        self.register_buffer("assigned_indicator_1", torch.zeros(self.n_choices))
        self.register_buffer("assigned_indicator_2", torch.zeros(self.n_choices))
        self.conv1_bops = 0
        self.conv2_bops = 0
        self.conv3_bops = 0
        self.indicator_1 = None
        self.indicator_2 = None
        self.init_state = False
        self.output_h = []
        self.output_w = []

    def get_bops(self):
        num_group_1 = self.indicator_1.sum()
        num_group_2 = self.indicator_2.sum()
        num_channels_1 = num_group_1 * self.group_size
        num_channels_2 = num_group_2 * self.group_size
        conv1_bops = compute_bops(
            self.conv1.kernel_size[0],
            self.conv1.in_channels,
            num_channels_1,
            self.output_h[0],
            self.output_w[0],
        )
        conv2_bops = compute_bops(
            self.conv2.kernel_size[0],
            num_channels_1,
            num_channels_2,
            self.output_h[1],
            self.output_w[1],
        )
        conv3_bops = compute_bops(
            self.conv2.kernel_size[0],
            num_channels_2,
            self.conv3.out_channels,
            self.output_h[1],
            self.output_w[1],
        )
        downsample_bops = 0
        if self.downsample is not None:
            downsample_bops = compute_bops(
                self.downsample[0].kernel_size[0],
                self.downsample[0].in_channels,
                self.downsample[0].out_channels,
                self.downsample[0].h,
                self.downsample[0].w,
            )
        total_bops = conv1_bops + conv2_bops + conv3_bops + downsample_bops
        return total_bops

    def get_total_bops(self):
        conv1_bops = compute_bops(
            self.conv1.kernel_size[0],
            self.conv1.in_channels,
            self.conv1.out_channels,
            self.output_h[0],
            self.output_w[0],
        )
        conv2_bops = compute_bops(
            self.conv2.kernel_size[0],
            self.conv2.in_channels,
            self.conv2.out_channels,
            self.output_h[1],
            self.output_w[1],
        )
        conv3_bops = compute_bops(
            self.conv3.kernel_size[0],
            self.conv3.in_channels,
            self.conv3.out_channels,
            self.output_h[2],
            self.output_w[2],
        )
        downsample_bops = 0
        if self.downsample is not None:
            downsample_bops = compute_bops(
                self.downsample[0].kernel_size[0],
                self.downsample[0].in_channels,
                self.downsample[0].out_channels,
                self.downsample[0].h,
                self.downsample[0].w,
            )
        total_bops = conv1_bops + conv2_bops + conv3_bops + downsample_bops
        return total_bops

    def compute_norm(self, filter_weight):
        n, c, h, w = filter_weight.shape
        filter_weight = filter_weight.reshape(
            n // self.group_size, self.group_size * c * h * w
        )
        normalized_filter_weight = filter_weight / torch.max(torch.abs(filter_weight))
        # l2-norm
        # normalized_filter_weight_norm = (normalized_filter_weight * normalized_filter_weight).sum(1) / (self.group_size * c * h * w)
        # l1-norm
        normalized_filter_weight_norm = normalized_filter_weight.abs().sum(1) / (
            self.group_size * c * h * w
        )
        # normalized_filter_weight_norm = (normalized_filter_weight.abs().sum(1) / (self.group_size * c * h * w)).detach()
        return normalized_filter_weight_norm

    def compute_indicator(self):
        # TODO: check whether to compute gradient with filter
        # TODO: check l1-norm and l2-norm
        # TODO: whether to use gradient scale, need to check on ImageNet
        # TODO: add maximum pruning bound
        filter_weight_1 = self.conv1.weight
        normalized_filter_weight_norm_1 = self.compute_norm(filter_weight_1)
        # grad_scale = 1 / math.sqrt(self.n_choices)
        # threshold = scale_grad(self.channel_thresholds, grad_scale)
        threshold_1 = self.channel_thresholds_1
        self.indicator_1 = (
            (normalized_filter_weight_norm_1 > threshold_1).float()
            - torch.sigmoid(normalized_filter_weight_norm_1 - threshold_1).detach()
            + torch.sigmoid(normalized_filter_weight_norm_1 - threshold_1)
        )
        self.assigned_indicator_1.data = self.indicator_1.data

        filter_weight_2 = self.conv2.weight
        normalized_filter_weight_norm_2 = self.compute_norm(filter_weight_2)
        # grad_scale = 1 / math.sqrt(self.n_choices)
        # threshold = scale_grad(self.channel_thresholds, grad_scale)
        threshold_2 = self.channel_thresholds_2
        self.indicator_2 = (
            (normalized_filter_weight_norm_2 > threshold_2).float()
            - torch.sigmoid(normalized_filter_weight_norm_2 - threshold_2).detach()
            + torch.sigmoid(normalized_filter_weight_norm_2 - threshold_2)
        )
        self.assigned_indicator_2.data = self.indicator_2.data
        return self.indicator_1, self.indicator_2

    def forward(self, x):
        if not self.init_state:
            residual = x

            out = self.conv1(x)
            _, _, conv1_out_shape_h, conv1_out_shape_w = out.shape
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            _, _, conv2_out_shape_h, conv2_out_shape_w = out.shape
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            _, _, conv3_out_shape_h, conv3_out_shape_w = out.shape
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            self.output_h.append(conv1_out_shape_h)
            self.output_h.append(conv2_out_shape_h)
            self.output_h.append(conv3_out_shape_h)
            self.output_w.append(conv1_out_shape_w)
            self.output_w.append(conv2_out_shape_w)
            self.output_w.append(conv3_out_shape_w)

            out += residual

            out = self.relu(out)
            self.init_state = True
        else:
            indicator_1, indicator_2 = self.compute_indicator()
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            n, c, h, w = out.shape
            reshape_out = out.reshape(n, c // self.group_size, self.group_size, h, w)
            reshape_out = reshape_out * indicator_1.unsqueeze(0).unsqueeze(
                -1
            ).unsqueeze(-1).unsqueeze(-1)
            out = reshape_out.reshape(n, c, h, w)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            n, c, h, w = out.shape
            reshape_out = out.reshape(n, c // self.group_size, self.group_size, h, w)
            reshape_out = reshape_out * indicator_2.unsqueeze(0).unsqueeze(
                -1
            ).unsqueeze(-1).unsqueeze(-1)
            out = reshape_out.reshape(n, c, h, w)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual

            out = self.relu(out)
            self.init_state = True

        return out


class SuperPrunedGateResNet(nn.Module):
    def __init__(self, depth, num_classes=1000):
        self.inplanes = 64
        super(SuperPrunedGateResNet, self).__init__()
        if depth < 50:
            block = SuperPrunedGateBasicBlock
        else:
            block = SuperPrunedGateBottleneck

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

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_bops(self):
        current_bops = 0
        layer = self.conv1
        bops = compute_bops(layer.kernel_size[0], layer.in_channels, layer.out_channels, layer.h, layer.w)
        current_bops += bops

        for i in range(4):
            layer = getattr(self, "layer{}".format(i + 1))
            for name, module in layer.named_modules():
                if isinstance(module, SuperPrunedGateBasicBlock):
                    bops = module.get_bops()
                    current_bops += bops
        
        layer = self.fc
        bops = compute_bops(1, layer.in_features, layer.out_features, 1, 1)
        current_bops += bops
        return current_bops
    
    def get_total_bops(self):
        current_bops = 0
        layer = self.conv1
        bops = compute_bops(layer.kernel_size[0], layer.in_channels, layer.out_channels, layer.h, layer.w)
        current_bops += bops

        for i in range(4):
            layer = getattr(self, "layer{}".format(i + 1))
            for name, module in layer.named_modules():
                if isinstance(module, (SuperPrunedGateBasicBlock, SuperPrunedGateBottleneck)):
                    bops = module.get_total_bops()
                    current_bops += bops

        layer = self.fc
        bops = compute_bops(1, layer.in_features, layer.out_features, 1, 1)
        current_bops += bops
        return current_bops

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
