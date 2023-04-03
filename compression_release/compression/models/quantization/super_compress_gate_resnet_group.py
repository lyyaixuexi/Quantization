# copy from pytorch-torchvision-models-resnet
import torch
import torch.nn as nn
from compression.models.quantization.dorefa_clip import QConv2d, QLinear
from compression.models.quantization.super_quan_conv import SuperQConv2d
from compression.utils.utils import *

__all__ = [
    "SuperCompressedGateResNetGroup",
    "SuperCompressedGateBasicBlockGroup",
    "SuperCompressedGateBottleneckGroup",
]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def superqconv3x3(
    in_planes,
    out_planes,
    stride=1,
    bits_weights_list=[2, 4, 8],
    bits_activations_list=[2, 4, 8],
):
    "3x3 convolution with padding"
    return SuperQConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        bits_weights_list=bits_weights_list,
        bits_activations_list=bits_activations_list,
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


class SuperCompressedGateBasicBlockGroup(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        group_size=8,
        bits_weights_list=[2, 4, 8],
        bits_activations_list=[2, 4, 8],
        # num_choices=8
    ):
        super(SuperCompressedGateBasicBlockGroup, self).__init__()
        self.name = "resnet-basic"
        self.conv1 = superqconv3x3(
            inplanes,
            planes,
            stride,
            bits_weights_list=bits_weights_list,
            bits_activations_list=bits_activations_list,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = superqconv3x3(
            planes,
            planes,
            bits_weights_list=bits_weights_list,
            bits_activations_list=bits_activations_list,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.block_index = 0

        self.group_size = group_size
        # self.num_choices = num_choices
        # self.group_size = planes // self.num_choices
        self.n_choices = planes // self.group_size
        self.channel_thresholds = nn.Parameter(torch.zeros(self.n_choices))
        self.register_buffer("assigned_indicator", torch.zeros(self.n_choices))
        self.indicator = None
        self.init_state = False
        self.output_h = []
        self.output_w = []

    def get_downsample_bops(self):
        downsample_bops = 0
        if self.downsample is not None:
            downsample_bops = compute_bops(
                self.downsample[0].kernel_size[0],
                self.downsample[0].in_channels,
                self.downsample[0].out_channels,
                self.downsample[0].h,
                self.downsample[0].w,
            )
        return downsample_bops

    def compress_fix_activation_compute_weight_bops(self):
        num_group = self.indicator.sum()
        channel_num = num_group * self.group_size
        # bops of conv1
        conv1_bops = self.conv1.compress_fix_activation_compute_weight_bops(
            channel_num, is_out_channel=True
        )
        # bops of conv2
        conv2_bops = self.conv2.compress_fix_activation_compute_weight_bops(
            channel_num, is_out_channel=False
        )
        # bops of downsample
        # downsample_bops = self.get_downsample_bops()
        downsample_bops = 0
        if self.downsample is not None:
            downsample_bops = self.downsample[0].fix_activation_compute_weight_bops()
        total_bops = conv1_bops + conv2_bops + downsample_bops
        return total_bops

    def compress_fix_weight_compute_activation_bops(self):
        num_group = self.indicator.sum()
        channel_num = num_group * self.group_size
        # bops of conv1
        conv1_bops = self.conv1.compress_fix_weight_compute_activation_bops(
            channel_num, is_out_channel=True
        )
        # bops of conv2
        conv2_bops = self.conv2.compress_fix_weight_compute_activation_bops(
            channel_num, is_out_channel=False
        )
        # downsample_bops = self.get_downsample_bops()
        downsample_bops = 0
        if self.downsample is not None:
            downsample_bops = self.downsample[0].fix_weight_compute_activation_bops()
        total_bops = conv1_bops + conv2_bops + downsample_bops
        return total_bops

    def compute_bops(self):
        num_group = self.indicator.sum()
        channel_num = num_group * self.group_size
        # bops of conv1
        conv1_bops = self.conv1.compute_current_bops(channel_num, is_out_channel=True)
        # bops of conv2
        conv2_bops = self.conv2.compute_current_bops(channel_num, is_out_channel=False)
        downsample_bops = self.get_downsample_bops()
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
        downsample_bops = self.get_downsample_bops()
        total_bops = conv1_bops + conv2_bops + downsample_bops
        return total_bops

    def compute_norm(self, filter_weight):
        n, c, h, w = filter_weight.shape
        filter_weight = filter_weight.reshape(
            n // self.group_size, self.group_size * c * h * w
        )
        # range [-1, 1]
        normalized_filter_weight = filter_weight / torch.max(torch.abs(filter_weight))
        # l1-norm
        normalized_filter_weight_norm = normalized_filter_weight.abs().sum(1) / (
            self.group_size * c * h * w
        )
        return normalized_filter_weight_norm

    def compute_indicator(self):
        # TODO: check whether to compute gradient with filter
        # TODO: check l1-norm and l2-norm
        # TODO: whether to use gradient scale, need to check on ImageNet
        # TODO: add maximum pruning bound
        # TODO: whether to use quantized filter?
        filter_weight = self.conv1.weight
        # quantized_weight = normalize_and_quantize_weight(filter_weight, self.conv1.bits_weights, self.conv1.weight_clip_value.detach())
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
            out = self.bn1(out)
            out = self.relu(out)

            _, _, conv1_out_shape_h, conv1_out_shape_w = out.shape

            out = self.conv2(out)
            out = self.bn2(out)

            _, _, conv2_out_shape_h, conv2_out_shape_w = out.shape

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)
            self.output_h.append(conv1_out_shape_h)
            self.output_h.append(conv2_out_shape_h)
            self.output_w.append(conv1_out_shape_w)
            self.output_w.append(conv2_out_shape_w)
            self.init_state = True
        else:
            indicator = self.compute_indicator()
            residual = x

            n, _, _, _ = self.conv1.weight.shape
            indicator = indicator.reshape(-1, 1)
            indicator = indicator.expand(n // self.group_size, self.group_size).reshape(
                n
            )
            index = (indicator > 0).nonzero().squeeze()
            selected_weight = torch.index_select(self.conv1.weight, 0, index)
            weight_mean = selected_weight.data.mean()
            weight_std = selected_weight.data.std()

            out = self.conv1(x, self.conv1.weight, weight_mean, weight_std)
            out = self.bn1(out)
            out = self.relu(out)

            selected_weight = torch.index_select(self.conv2.weight, 1, index)
            weight_mean = selected_weight.data.mean()
            weight_std = selected_weight.data.std()
            out = out * indicator.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            out = self.conv2(out, self.conv2.weight, weight_mean, weight_std)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

        return out


class SuperCompressedGateBottleneckGroup(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        # num_choices=8
        group_size=8,
        bits_weights_list=[2, 4, 8],
        bits_activations_list=[2, 4, 8],
    ):
        super(SuperCompressedGateBottleneckGroup, self).__init__()
        self.name = "resnet-bottleneck"
        self.conv1 = SuperQConv2d(
            inplanes,
            planes,
            kernel_size=1,
            bias=False,
            bits_weights_list=bits_weights_list,
            bits_activations_list=bits_activations_list,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SuperQConv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            bits_weights_list=bits_weights_list,
            bits_activations_list=bits_activations_list,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SuperQConv2d(
            planes,
            planes * 4,
            kernel_size=1,
            bias=False,
            bits_weights_list=bits_weights_list,
            bits_activations_list=bits_activations_list,
        )
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.block_index = 0

        # self.num_choices = num_choices
        # self.group_size = planes // self.num_choices
        self.group_size = group_size
        self.n_choices = planes // self.group_size
        self.channel_thresholds_1 = nn.Parameter(torch.zeros(self.n_choices))
        self.register_buffer("assigned_indicator_1", torch.zeros(self.n_choices))
        self.indicator_1 = None
        self.channel_thresholds_2 = nn.Parameter(torch.zeros(self.n_choices))
        self.register_buffer("assigned_indicator_2", torch.zeros(self.n_choices))
        self.indicator_2 = None
        self.init_state = False
        self.output_h = []
        self.output_w = []

    def get_downsample_bops(self):
        downsample_bops = 0
        if self.downsample is not None:
            downsample_bops = compute_bops(
                self.downsample[0].kernel_size[0],
                self.downsample[0].in_channels,
                self.downsample[0].out_channels,
                self.downsample[0].h,
                self.downsample[0].w,
            )
        return downsample_bops

    def compress_fix_activation_compute_weight_bops(self):
        num_group_1 = self.indicator_1.sum()
        channel_num_1 = num_group_1 * self.group_size
        num_group_2 = self.indicator_2.sum()
        channel_num_2 = num_group_2 * self.group_size
        # bops of conv1
        conv1_bops = self.conv1.compress_fix_activation_compute_weight_bops(
            channel_num_1, is_out_channel=True
        )
        # bops of conv2
        conv2_bops = self.conv2.compress_channel_fix_activation_compute_weight_bops(
            channel_num_1, channel_num_2
        )
        # bops of conv3
        conv3_bops = self.conv3.compress_fix_activation_compute_weight_bops(
            channel_num_2, is_out_channel=False
        )
        # bops of downsample
        # downsample_bops = self.get_downsample_bops()
        downsample_bops = 0
        if self.downsample is not None:
            downsample_bops = self.downsample[0].fix_activation_compute_weight_bops()
        total_bops = conv1_bops + conv2_bops + conv3_bops + downsample_bops
        return total_bops

    def compress_fix_weight_compute_activation_bops(self):
        num_group_1 = self.indicator_1.sum()
        channel_num_1 = num_group_1 * self.group_size
        num_group_2 = self.indicator_2.sum()
        channel_num_2 = num_group_2 * self.group_size
        # bops of conv1
        conv1_bops = self.conv1.compress_fix_weight_compute_activation_bops(
            channel_num_1, is_out_channel=True
        )
        # bops of conv2
        conv2_bops = self.conv2.compress_channel_fix_weight_compute_activation_bops(
            channel_num_1, channel_num_2
        )
        # bops of conv2
        conv3_bops = self.conv3.compress_fix_weight_compute_activation_bops(
            channel_num_2, is_out_channel=False
        )
        # downsample_bops = self.get_downsample_bops()
        downsample_bops = 0
        if self.downsample is not None:
            downsample_bops = self.downsample[0].fix_weight_compute_activation_bops()
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
        downsample_bops = self.get_downsample_bops()
        total_bops = conv1_bops + conv2_bops + conv3_bops + downsample_bops
        return total_bops

    def compute_norm(self, filter_weight):
        n, c, h, w = filter_weight.shape
        filter_weight = filter_weight.reshape(
            n // self.group_size, self.group_size * c * h * w
        )
        # range [-1, 1]
        normalized_filter_weight = filter_weight / torch.max(torch.abs(filter_weight))
        # l1-norm
        normalized_filter_weight_norm = normalized_filter_weight.abs().sum(1) / (
            self.group_size * c * h * w
        )
        return normalized_filter_weight_norm

    def compute_indicator(self):
        # TODO: check whether to compute gradient with filter
        # TODO: check l1-norm and l2-norm
        # TODO: whether to use gradient scale, need to check on ImageNet
        # TODO: add maximum pruning bound
        # TODO: whether to use quantized filter?
        filter_weight_1 = self.conv1.weight
        normalized_filter_weight_norm_1 = self.compute_norm(filter_weight_1)
        threshold = self.channel_thresholds_1
        self.indicator_1 = (
            (normalized_filter_weight_norm_1 > threshold).float()
            - torch.sigmoid(normalized_filter_weight_norm_1 - threshold).detach()
            + torch.sigmoid(normalized_filter_weight_norm_1 - threshold)
        )
        self.assigned_indicator_1.data = self.indicator_1.data

        filter_weight_2 = self.conv2.weight
        normalized_filter_weight_norm_2 = self.compute_norm(filter_weight_2)
        threshold = self.channel_thresholds_2
        self.indicator_2 = (
            (normalized_filter_weight_norm_2 > threshold).float()
            - torch.sigmoid(normalized_filter_weight_norm_2 - threshold).detach()
            + torch.sigmoid(normalized_filter_weight_norm_2 - threshold)
        )
        self.assigned_indicator_2.data = self.indicator_2.data
        return self.indicator_1, self.indicator_2

    def forward(self, x):
        if not self.init_state:
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            _, _, conv1_out_shape_h, conv1_out_shape_w = out.shape

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            _, _, conv2_out_shape_h, conv2_out_shape_w = out.shape

            out = self.conv3(out)
            out = self.bn3(out)

            _, _, conv3_out_shape_h, conv3_out_shape_w = out.shape

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual

            out = self.relu(out)
            self.output_h.append(conv1_out_shape_h)
            self.output_h.append(conv2_out_shape_h)
            self.output_h.append(conv3_out_shape_h)
            self.output_w.append(conv1_out_shape_w)
            self.output_w.append(conv2_out_shape_w)
            self.output_w.append(conv3_out_shape_w)
            self.init_state = True
        else:
            indicator_1, indicator_2 = self.compute_indicator()
            residual = x

            n, _, _, _ = self.conv1.weight.shape
            indicator_1 = indicator_1.reshape(-1, 1)
            indicator_1 = indicator_1.expand(
                n // self.group_size, self.group_size
            ).reshape(n)
            index_1 = (indicator_1 > 0).nonzero().squeeze()
            selected_weight = torch.index_select(self.conv1.weight, 0, index_1)
            weight_mean = selected_weight.data.mean()
            weight_std = selected_weight.data.std()

            out = self.conv1(x, self.conv1.weight, weight_mean, weight_std)
            out = self.bn1(out)
            out = self.relu(out)

            n, _, _, _ = self.conv2.weight.shape
            indicator_2 = indicator_2.reshape(-1, 1)
            indicator_2 = indicator_2.expand(
                n // self.group_size, self.group_size
            ).reshape(n)
            index_2 = (indicator_2 > 0).nonzero().squeeze()
            selected_weight = torch.index_select(self.conv2.weight, 1, index_1)
            selected_weight = torch.index_select(selected_weight, 0, index_2)
            weight_mean = selected_weight.data.mean()
            weight_std = selected_weight.data.std()

            out = out * indicator_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            out = self.conv2(out, self.conv2.weight, weight_mean, weight_std)
            out = self.bn2(out)
            out = self.relu(out)

            out = out * indicator_2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            selected_weight = torch.index_select(self.conv3.weight, 1, index_2)
            weight_mean = selected_weight.data.mean()
            weight_std = selected_weight.data.std()

            out = self.conv3(out, self.conv3.weight, weight_mean, weight_std)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual

            out = self.relu(out)
            self.init_state = True

        return out


class SuperCompressedGateResNetGroup(nn.Module):
    def __init__(
        self,
        depth,
        num_classes=1000,
        group_size=8,
        bits_weights_list=[2, 4, 8],
        bits_activations_list=[2, 4, 8],
    ):
        self.inplanes = 64
        super(SuperCompressedGateResNetGroup, self).__init__()
        if depth < 50:
            block = SuperCompressedGateBasicBlockGroup
        else:
            block = SuperCompressedGateBottleneckGroup

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
        self.conv1 = QConv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            bits_weights=8,
            bits_activations=32,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block,
            64,
            layers[0],
            # num_choices=num_choices
            group_size=group_size,
            bits_weights_list=bits_weights_list,
            bits_activations_list=bits_activations_list,
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            # num_choices=num_choices,
            group_size=group_size,
            bits_weights_list=bits_weights_list,
            bits_activations_list=bits_activations_list,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            # num_choices=num_choices,
            group_size=group_size,
            bits_weights_list=bits_weights_list,
            bits_activations_list=bits_activations_list,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            # num_choices=num_choices,
            group_size=group_size,
            bits_weights_list=bits_weights_list,
            bits_activations_list=bits_activations_list,
        )
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = QLinear(
            512 * block.expansion, num_classes, bits_weights=8, bits_activations=8
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        group_size=8,
        bits_weights_list=[2, 4, 8],
        bits_activations_list=[2, 4, 8],
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SuperQConv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    bits_weights_list=bits_weights_list,
                    bits_activations_list=bits_activations_list,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                group_size=group_size,
                bits_weights_list=bits_weights_list,
                bits_activations_list=bits_activations_list,
                # num_choices=num_choices,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    group_size=group_size,
                    bits_weights_list=bits_weights_list,
                    bits_activations_list=bits_activations_list,
                    # num_choices=num_choices,
                )
            )

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
