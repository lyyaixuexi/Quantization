import logging

import torch
import torch.nn as nn
from compression.models.quantization import dorefa_clip, dorefa_clip_asymmetric
from torch.nn import functional as F

__all__ = ["ResModelPrune", "get_select_channels"]

logger = logging.getLogger("baseline")


def get_select_channels(d):
    """
    Get select channels
    """

    select_channels = (d > 0).nonzero().squeeze()
    return select_channels


def get_thin_params(layer, select_channels, dim=0):
    """
    Get params from layers after pruning
    """

    if isinstance(layer, nn.Conv2d):
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        if layer.bias is not None:
            if dim == 0:
                thin_bias = layer.bias.data.index_select(dim, select_channels)
            else:
                thin_bias = layer.bias.data
        else:
            thin_bias = None
        if hasattr(layer, "weight_clip_value"):
            weight_clip_value = layer.weight_clip_value.data
            activation_clip_value = layer.activation_clip_value.data
            if hasattr(layer, "activation_bias"):
                activation_bias = layer.activation_bias
                thin_weight = (
                    thin_weight,
                    weight_clip_value,
                    activation_clip_value,
                    activation_bias,
                )
            else:
                thin_weight = (thin_weight, weight_clip_value, activation_clip_value)

    elif isinstance(layer, nn.BatchNorm2d):
        assert dim == 0, "invalid dimension for bn_layer"

        thin_weight = layer.weight.data.index_select(dim, select_channels)
        thin_mean = layer.running_mean.index_select(dim, select_channels)
        thin_var = layer.running_var.index_select(dim, select_channels)
        if layer.bias is not None:
            thin_bias = layer.bias.data.index_select(dim, select_channels)
        else:
            thin_bias = None
        return (thin_weight, thin_mean), (thin_bias, thin_var)
    elif isinstance(layer, nn.PReLU):
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        thin_bias = None

    elif isinstance(layer, nn.Linear):
        # assert dim == 1, "invalid dimension for linear layer"
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        if hasattr(layer, "weight_clip_value"):
            weight_clip_value = layer.weight_clip_value.data
            activation_clip_value = layer.activation_clip_value.data
            if hasattr(layer, "activation_bias"):
                activation_bias = layer.activation_bias
                thin_weight = (
                    thin_weight,
                    weight_clip_value,
                    activation_clip_value,
                    activation_bias,
                )
            else:
                thin_weight = (thin_weight, weight_clip_value, activation_clip_value)
        if dim == 1:
            thin_bias = layer.bias.data
        else:
            thin_bias = layer.bias.data.index_select(0, select_channels)

    return thin_weight, thin_bias


def replace_layer(old_layer, init_weight, init_bias=None):
    """
    Replace specific layer of model
    :params layer: original layer
    :params init_weight: thin_weight
    :params init_bias: thin_bias
    :params keeping: whether to keep MaskConv2d
    """

    if hasattr(old_layer, "bias") and old_layer.bias is not None:
        bias_flag = True
    else:
        bias_flag = False
    if isinstance(old_layer, (dorefa_clip.QConv2d, dorefa_clip_asymmetric.QConv2d)):
        weight = init_weight[0]
        weight_clip_value = init_weight[1]
        activation_clip_value = init_weight[2]
        if isinstance(old_layer, dorefa_clip_asymmetric.QConv2d):
            activation_bias = init_weight[3]

        if old_layer.groups != 1:
            new_groups = weight.size(0)
            in_channels = weight.size(0)
            out_channels = weight.size(0)
        else:
            new_groups = 1
            in_channels = weight.size(1)
            out_channels = weight.size(0)
        if isinstance(old_layer, dorefa_clip.QConv2d):
            new_layer = dorefa_clip.QConv2d(
                in_channels,
                out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=bias_flag,
                bits_weights=old_layer.bits_weights,
                bits_activations=old_layer.bits_activations,
                groups=new_groups,
            )
        else:
            new_layer = dorefa_clip_asymmetric.QConv2d(
                in_channels,
                out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=bias_flag,
                bits_weights=old_layer.bits_weights,
                bits_activations=old_layer.bits_activations,
                groups=new_groups,
            )

        new_layer.weight.data.copy_(weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)
        new_layer.weight_clip_value.data.copy_(weight_clip_value)
        new_layer.activation_clip_value.data.copy_(activation_clip_value)
        if isinstance(old_layer, dorefa_clip_asymmetric.QConv2d):
            new_layer.activation_bias.data.copy_(activation_bias)

    elif isinstance(old_layer, nn.Conv2d):
        weight = init_weight
        if old_layer.groups != 1:
            new_groups = weight.size(0)
            in_channels = weight.size(0)
            out_channels = weight.size(0)
        else:
            new_groups = 1
            in_channels = weight.size(1)
            out_channels = weight.size(0)
        new_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=old_layer.kernel_size,
            stride=old_layer.stride,
            padding=old_layer.padding,
            bias=bias_flag,
            groups=new_groups,
        )
        new_layer.weight.data.copy_(weight)
        if init_bias is not None:
            new_layer.bias.data.copy_(init_bias)

    elif isinstance(old_layer, nn.BatchNorm2d):
        weight = init_weight[0]
        mean_ = init_weight[1]
        bias = init_bias[0]
        var_ = init_bias[1]
        new_layer = nn.BatchNorm2d(weight.size(0))
        new_layer.weight.data.copy_(weight)
        assert init_bias is not None, "batch normalization needs bias"
        new_layer.bias.data.copy_(bias)
        new_layer.running_mean.copy_(mean_)
        new_layer.running_var.copy_(var_)
    elif isinstance(old_layer, nn.PReLU):
        new_layer = nn.PReLU(init_weight.size(0))
        new_layer.weight.data.copy_(init_weight)
    elif isinstance(old_layer, dorefa_clip_asymmetric.QLinear):
        weight = init_weight[0]
        weight_clip_value = init_weight[1]
        activation_clip_value = init_weight[2]
        activation_bias = init_weight[3]
        new_layer = dorefa_clip_asymmetric.QLinear(
            in_features=weight.shape[1],
            out_features=weight.shape[0],
            bits_weights=old_layer.bits_weights,
            bits_activations=old_layer.bits_activations,
        )
        new_layer.weight.data.copy_(weight)
        new_layer.bias.data.copy_(init_bias)

        new_layer.weight_clip_value.data.copy_(weight_clip_value)
        new_layer.activation_clip_value.data.copy_(activation_clip_value)
        new_layer.activation_bias.data.copy_(activation_bias)
    elif isinstance(old_layer, nn.Linear):
        new_layer = nn.Linear(
            in_features=init_weight.shape[1], out_features=init_weight.shape[0]
        )
        new_layer.weight.data.copy_(init_weight)
        new_layer.bias.data.copy_(init_bias)
    else:
        assert False, "unsupport layer type:" + str(type(old_layer))
    return new_layer


class ResBlockPrune(object):
    """
    Residual block pruning
    """

    def __init__(self, block, block_type):
        self.block = block
        self.block_type = block_type
        self.select_channels = None

    def pruning_preresnet(self):
        # compute selected channels
        select_channels = get_select_channels(self.block.conv1.mask)
        self.select_channels = select_channels

        # prune and replace conv1
        thin_weight, thin_bias = get_thin_params(self.block.conv1, select_channels, 0)
        self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)

        # prune and replace bn2
        thin_weight, thin_bias = get_thin_params(self.block.bn2, select_channels, 0)
        self.block.bn2 = replace_layer(self.block.bn2, thin_weight, thin_bias)

        # prune and replace conv2
        thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 1)
        self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

        if self.block.downsample is not None:
            thin_weight = self.block.downsample.weight
            if self.block.downsample.bias is not None:
                thin_bias = self.block.downsample.bias
            else:
                thin_bias = None
            weight_clip_value = self.block.downsample.weight_clip_value.data
            activation_clip_value = self.block.downsample.activation_clip_value
            thin_weight = (thin_weight, weight_clip_value, activation_clip_value)
            self.block.downsample = replace_layer(
                self.block.downsample, thin_weight, thin_bias
            )

    def pruning_scale_preresnet(self):
        # compute selected channels
        select_channels = get_select_channels(self.block.choices_path_weight)
        self.select_channels = select_channels

        # prune and replace conv1
        thin_weight, thin_bias = get_thin_params(self.block.conv1, select_channels, 0)
        print(thin_weight)
        self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)

        # prune and replace bn2
        thin_weight, thin_bias = get_thin_params(self.block.bn2, select_channels, 0)
        self.block.bn2 = replace_layer(self.block.bn2, thin_weight, thin_bias)

        # prune and replace conv2
        thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 1)
        self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

    def pruning_gate_preresnet(self):
        # channel_thresholds = self.block.channel_thresholds
        # filter_weight = self.block.conv1.weight
        # group_size = self.block.group_size
        # n, c, h, w = filter_weight.shape
        # filter_weight = filter_weight.reshape(n // group_size, group_size * c * h * w)
        # normalized_filter_weight = filter_weight / torch.max(torch.abs(filter_weight))
        # # l2-norm
        # # normalized_filter_weight_norm = (normalized_filter_weight * normalized_filter_weight).sum(1) / (group_size * c * h * w)
        # # l1-norm
        # normalized_filter_weight_norm = normalized_filter_weight.abs().sum(1) / (group_size * c * h * w)
        # indicator = (normalized_filter_weight_norm > channel_thresholds).float().reshape(-1, 1)
        # indicator = indicator.expand(n // group_size, group_size).reshape(n)
        filter_weight = self.block.conv1.weight
        n, c, h, w = filter_weight.shape
        group_size = self.block.group_size
        indicator = self.block.assigned_indicator.reshape(-1, 1)
        indicator = indicator.expand(n // group_size, group_size).reshape(n)

        # compute selected channels
        select_channels = get_select_channels(indicator)
        self.select_channels = select_channels

        # prune and replace conv1
        thin_weight, thin_bias = get_thin_params(self.block.conv1, select_channels, 0)
        self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)

        # prune and replace bn2
        thin_weight, thin_bias = get_thin_params(self.block.bn2, select_channels, 0)
        self.block.bn2 = replace_layer(self.block.bn2, thin_weight, thin_bias)

        # prune and replace conv2
        thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 1)
        self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

    def pruning_gate_resnet(self):
        filter_weight = self.block.conv1.weight
        n, c, h, w = filter_weight.shape
        group_size = self.block.group_size
        indicator = self.block.assigned_indicator.reshape(-1, 1)
        indicator = indicator.expand(n // group_size, group_size).reshape(n)

        # compute selected channels
        select_channels = get_select_channels(indicator)
        self.select_channels = select_channels

        # prune and replace conv1
        thin_weight, thin_bias = get_thin_params(self.block.conv1, select_channels, 0)
        self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)

        # prune and replace bn1
        thin_weight, thin_bias = get_thin_params(self.block.bn1, select_channels, 0)
        self.block.bn1 = replace_layer(self.block.bn1, thin_weight, thin_bias)

        # prune and replace conv2
        thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 1)
        self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

    def pruning_gate_resnet_bottleneck(self):
        filter_weight = self.block.conv1.weight
        n, c, h, w = filter_weight.shape
        group_size = self.block.group_size
        indicator = self.block.assigned_indicator_1.reshape(-1, 1)
        indicator = indicator.expand(n // group_size, group_size).reshape(n)

        # compute selected channels
        select_channels = get_select_channels(indicator)
        self.select_channels = select_channels

        # prune and replace conv1
        thin_weight, thin_bias = get_thin_params(self.block.conv1, select_channels, 0)
        self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)

        # prune and replace bn1
        thin_weight, thin_bias = get_thin_params(self.block.bn1, select_channels, 0)
        self.block.bn1 = replace_layer(self.block.bn1, thin_weight, thin_bias)

        # prune and replace conv2
        thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 1)
        self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

        filter_weight = self.block.conv2.weight
        n, c, h, w = filter_weight.shape
        group_size = self.block.group_size
        indicator = self.block.assigned_indicator_2.reshape(-1, 1)
        indicator = indicator.expand(n // group_size, group_size).reshape(n)

        # prune and replace conv2
        thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 0)
        self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

        # prune and replace bn2
        thin_weight, thin_bias = get_thin_params(self.block.bn2, select_channels, 0)
        self.block.bn2 = replace_layer(self.block.bn2, thin_weight, thin_bias)

        # prune and replace conv3
        thin_weight, thin_bias = get_thin_params(self.block.conv3, select_channels, 1)
        self.block.conv3 = replace_layer(self.block.conv3, thin_weight, thin_bias)

        # print(self.block.conv1)
        # print(self.block.conv2)

    def pruning_gate_mobilenetv2(self):
        filter_weight = self.block.conv2.weight
        n, c, h, w = filter_weight.shape
        group_size = self.block.group_size
        indicator = self.block.assigned_indicator.reshape(-1, 1)
        indicator = indicator.expand(n // group_size, group_size).reshape(n)

        # compute selected channels
        select_channels = get_select_channels(indicator)
        self.select_channels = select_channels

        # prune and replace conv1
        if hasattr(self.block, "conv1"):
            thin_weight, thin_bias = get_thin_params(
                self.block.conv1, select_channels, 0
            )
            self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)

            # prune and replace bn1
            thin_weight, thin_bias = get_thin_params(self.block.bn1, select_channels, 0)
            self.block.bn1 = replace_layer(self.block.bn1, thin_weight, thin_bias)

        # prune and replace conv2
        thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 0)
        self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

        # prune and replace bn2
        thin_weight, thin_bias = get_thin_params(self.block.bn2, select_channels, 0)
        self.block.bn2 = replace_layer(self.block.bn2, thin_weight, thin_bias)

        # prune and replace conv3
        thin_weight, thin_bias = get_thin_params(self.block.conv3, select_channels, 1)
        self.block.conv3 = replace_layer(self.block.conv3, thin_weight, thin_bias)

    def pruning_gate_mobilenetv3(self):
        filter_weight = self.block.conv2.weight
        n, c, h, w = filter_weight.shape
        group_size = self.block.group_size
        indicator = self.block.assigned_indicator.reshape(-1, 1)
        indicator = indicator.expand(n // group_size, group_size).reshape(n)

        # compute selected channels
        select_channels = get_select_channels(indicator)
        self.select_channels = select_channels

        # prune and replace conv1
        if hasattr(self.block, "conv1"):
            thin_weight, thin_bias = get_thin_params(
                self.block.conv1, select_channels, 0
            )
            self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)

            # prune and replace bn1
            thin_weight, thin_bias = get_thin_params(self.block.bn1, select_channels, 0)
            self.block.bn1 = replace_layer(self.block.bn1, thin_weight, thin_bias)

        # prune and replace conv2
        thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 0)
        # print("Len of weight: {}".format(len(thin_weight)))
        # print(thin_weight.shape)
        self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

        # prune and replace bn2
        thin_weight, thin_bias = get_thin_params(self.block.bn2, select_channels, 0)
        self.block.bn2 = replace_layer(self.block.bn2, thin_weight, thin_bias)

        # prune and replace conv3
        thin_weight, thin_bias = get_thin_params(self.block.conv3, select_channels, 1)
        self.block.conv3 = replace_layer(self.block.conv3, thin_weight, thin_bias)

        if self.block.SE:
            # prune and replace se
            thin_weight, thin_bias = get_thin_params(
                self.block.squeeze_block.dense[0], select_channels, 1
            )
            self.block.squeeze_block.dense[0] = replace_layer(
                self.block.squeeze_block.dense[0], thin_weight, thin_bias
            )

            thin_weight, thin_bias = get_thin_params(
                self.block.squeeze_block.dense[2], select_channels, 0
            )
            self.block.squeeze_block.dense[2] = replace_layer(
                self.block.squeeze_block.dense[2], thin_weight, thin_bias
            )

    def pruning(self):
        """
        Perform pruning
        """

        # prune pre-resnet on cifar
        if self.block_type in ["super_preresnet"]:
            self.pruning_preresnet()
        elif self.block_type in ["preresnet", "qpreresnet"]:
            self.pruning_gate_preresnet()
        elif self.block_type in ["resnet_basic"]:
            self.pruning_gate_resnet()
        elif self.block_type in ["resnet_bottleneck"]:
            self.pruning_gate_resnet_bottleneck()
        elif self.block_type in ["mobilenetv2", "qmobilenetv2"]:
            self.pruning_gate_mobilenetv2()
        elif self.block_type in ["qasymobilenetv3_cifar", "mobilenetv3_cifar"]:
            self.pruning_gate_mobilenetv3()
        else:
            assert False, "invalid block type: " + self.block_type


class ResSeqPrune(object):
    """
    Sequantial pruning
    """

    def __init__(self, sequential, seq_type):
        self.sequential = sequential
        self.sequential_length = len(list(self.sequential))
        self.res_block_prune = []

        for i in range(self.sequential_length):
            self.res_block_prune.append(
                ResBlockPrune(self.sequential[i], block_type=seq_type)
            )

    def pruning(self):
        """
        Perform pruning
        """

        for i in range(self.sequential_length):
            self.res_block_prune[i].pruning()

        temp_seq = []
        for i in range(self.sequential_length):
            if self.res_block_prune[i].block is not None:
                temp_seq.append(self.res_block_prune[i].block)
        self.sequential = nn.Sequential(*temp_seq)


class ResModelPrune(object):
    """
    Prune residual networks
    """

    def __init__(self, model, net_type, depth):
        self.model = model
        if net_type in ["resnet", "qresnet"]:
            if depth >= 50:
                self.net_type = "resnet_bottleneck"
            else:
                self.net_type = "resnet_basic"
        else:
            self.net_type = net_type
        logger.info("|===>Init ResModelPrune")

    def run(self):
        """
        Perform pruning
        """
        if self.net_type in ["preresnet", "qpreresnet"]:
            res_seq_prune = [
                ResSeqPrune(self.model.layer1, seq_type=self.net_type),
                ResSeqPrune(self.model.layer2, seq_type=self.net_type),
                ResSeqPrune(self.model.layer3, seq_type=self.net_type),
            ]
            for i in range(3):
                res_seq_prune[i].pruning()

            self.model.layer1 = res_seq_prune[0].sequential
            self.model.layer2 = res_seq_prune[1].sequential
            self.model.layer3 = res_seq_prune[2].sequential
            # logger.info(self.model)

        elif self.net_type in ["resnet_basic", "resnet_bottleneck"]:
            res_seq_prune = [
                ResSeqPrune(self.model.layer1, seq_type=self.net_type),
                ResSeqPrune(self.model.layer2, seq_type=self.net_type),
                ResSeqPrune(self.model.layer3, seq_type=self.net_type),
                ResSeqPrune(self.model.layer4, seq_type=self.net_type),
            ]
            for i in range(4):
                res_seq_prune[i].pruning()

            self.model.layer1 = res_seq_prune[0].sequential
            self.model.layer2 = res_seq_prune[1].sequential
            self.model.layer3 = res_seq_prune[2].sequential
            self.model.layer4 = res_seq_prune[3].sequential

        elif self.net_type in ["mobilenetv2", "qmobilenetv2"]:
            res_seq_prune = [
                ResSeqPrune(self.model.layer1, seq_type=self.net_type),
                ResSeqPrune(self.model.layer2, seq_type=self.net_type),
                ResSeqPrune(self.model.layer3, seq_type=self.net_type),
                ResSeqPrune(self.model.layer4, seq_type=self.net_type),
                ResSeqPrune(self.model.layer5, seq_type=self.net_type),
                ResSeqPrune(self.model.layer6, seq_type=self.net_type),
                ResSeqPrune(self.model.layer7, seq_type=self.net_type),
            ]

            # prune the conv in the first sequence
            filter_weight = self.model.layer1[0].conv2.weight
            n, c, h, w = filter_weight.shape
            group_size = self.model.layer1[0].group_size
            indicator = self.model.layer1[0].assigned_indicator.reshape(-1, 1)
            indicator = indicator.expand(n // group_size, group_size).reshape(n)

            for i in range(7):
                res_seq_prune[i].pruning()
            self.model.layer1 = res_seq_prune[0].sequential
            self.model.layer2 = res_seq_prune[1].sequential
            self.model.layer3 = res_seq_prune[2].sequential
            self.model.layer4 = res_seq_prune[3].sequential
            self.model.layer5 = res_seq_prune[4].sequential
            self.model.layer6 = res_seq_prune[5].sequential
            self.model.layer7 = res_seq_prune[6].sequential

            # prune and replace conv1 in the network
            select_channels = get_select_channels(indicator)
            thin_weight, thin_bias = get_thin_params(
                self.model.conv1, select_channels, 0
            )
            self.model.conv1 = replace_layer(self.model.conv1, thin_weight, thin_bias)

            # prune and replace bn1 in the network
            thin_weight, thin_bias = get_thin_params(self.model.bn1, select_channels, 0)
            self.model.bn1 = replace_layer(self.model.bn1, thin_weight, thin_bias)

        elif self.net_type in ["qasymobilenetv3_cifar", "mobilenetv3_cifar"]:
            print("test")
            res_seq_prune = ResSeqPrune(self.model.block, seq_type=self.net_type)

            # prune the conv in the first sequence
            filter_weight = self.model.block[0].conv2.weight
            n, c, h, w = filter_weight.shape
            group_size = self.model.block[0].group_size
            indicator = self.model.block[0].assigned_indicator.reshape(-1, 1)
            indicator = indicator.expand(n // group_size, group_size).reshape(n)

            res_seq_prune.pruning()
            self.model.block = res_seq_prune.sequential

            # prune and replace conv1 in the network
            select_channels = get_select_channels(indicator)
            thin_weight, thin_bias = get_thin_params(
                self.model.conv1, select_channels, 0
            )
            self.model.conv1 = replace_layer(self.model.conv1, thin_weight, thin_bias)

            # prune and replace bn1 in the network
            thin_weight, thin_bias = get_thin_params(self.model.bn1, select_channels, 0)
            self.model.bn1 = replace_layer(self.model.bn1, thin_weight, thin_bias)
