import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from compression.models.quantization.dorefa_clip import QConv2d, QLinear
from compression.models.quantization.super_quan_conv import SuperQConv2d

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
        assert dim == 1, "invalid dimension for linear layer"
        thin_weight = layer.weight.data.index_select(dim, select_channels)
        thin_bias = layer.bias.data

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

    if isinstance(old_layer, nn.Conv2d):
        weight = init_weight
        new_layer = SuperQConv2d(
            weight.size(1),
            weight.size(0),
            kernel_size=old_layer.kernel_size,
            stride=old_layer.stride,
            padding=old_layer.padding,
            bias=bias_flag,
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
    elif isinstance(old_layer, nn.Linear):
        new_layer = nn.Linear(in_features=init_weight.shape[1], out_features=init_weight.shape[0])
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

    def pruning_gate_preresnet(self):
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

    def pruning(self):
        """
        Perform pruning
        """

        # prune pre-resnet on cifar
        if self.block_type in ["super_quan_preresnet"]:
            self.pruning_gate_preresnet()
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
        self.net_type = net_type
        logger.info("|===>Init ResModelPrune")

    def run(self):
        """
        Perform pruning
        """
        if self.net_type in ["super_quan_preresnet"]:
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
