import logging

import torch.nn as nn

from compression.utils.pruning import get_select_channels, get_thin_params, replace_layer

__all__ = ["ResModelPrune", "get_select_channels"]

logger = logging.getLogger("baseline")


class ResBlockPrune(object):
    """
    Residual block pruning
    """

    def __init__(self, block, current_block_indicator, previous_block_indicator, block_type):
        self.current_block_indicator = current_block_indicator
        self.previous_block_indicator = previous_block_indicator
        self.block = block
        self.block_type = block_type
        self.select_channels = None

    def pruning_gate_preresnet(self):
        filter_weight = self.block.conv1.weight
        n, _, _, _ = filter_weight.shape
        group_size = self.block.group_size
        indicator = self.block.assigned_indicator.reshape(-1, 1)
        indicator = indicator.expand(n // group_size, group_size).reshape(n)

        # prune and replace conv1 out channels
        select_channels = get_select_channels(indicator)
        self.select_channels = select_channels
        thin_weight, thin_bias = get_thin_params(self.block.conv1, select_channels, 0)
        self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)

        # prune and replace bn2
        thin_weight, thin_bias = get_thin_params(self.block.bn2, select_channels, 0)
        self.block.bn2 = replace_layer(self.block.bn2, thin_weight, thin_bias)

        # prune and replace conv2 in channels
        thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 1)
        self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

        # prune and replace conv1 in channels
        if self.block.downsample:
            block_indicator = self.previous_block_indicator
            filter_weight = self.block.conv1.weight
            _, n, _, _ =  filter_weight.shape
        else:
            block_indicator = self.current_block_indicator
            filter_weight = self.block.conv2.weight
            n, _, _, _ =  filter_weight.shape
        group_size = self.block.group_size
        block_indicator = block_indicator.reshape(-1, 1)
        block_indicator = block_indicator.expand(n // group_size, group_size).reshape(n)

        select_channels = get_select_channels(block_indicator)
        thin_weight, thin_bias = get_thin_params(self.block.conv1, select_channels, 1)
        self.block.conv1 = replace_layer(self.block.conv1, thin_weight, thin_bias)

        # prune and replace bn1
        thin_weight, thin_bias = get_thin_params(self.block.bn1, select_channels, 0)
        self.block.bn1 = replace_layer(self.block.bn1, thin_weight, thin_bias)

        if self.block.downsample:
            thin_weight, thin_bias = get_thin_params(self.block.downsample, select_channels, 1)
            self.block.downsample = replace_layer(self.block.downsample, thin_weight, thin_bias)

        # prune and replace conv2 out channels
        block_indicator = self.current_block_indicator
        filter_weight = self.block.conv2.weight
        n, _, _, _ =  filter_weight.shape
        group_size = self.block.group_size
        block_indicator = block_indicator.reshape(-1, 1)
        block_indicator = block_indicator.expand(n // group_size, group_size).reshape(n)

        select_channels = get_select_channels(block_indicator)
        thin_weight, thin_bias = get_thin_params(self.block.conv2, select_channels, 0)
        self.block.conv2 = replace_layer(self.block.conv2, thin_weight, thin_bias)

        if self.block.downsample:
            thin_weight, thin_bias = get_thin_params(self.block.downsample, select_channels, 0)
            self.block.downsample = replace_layer(self.block.downsample, thin_weight, thin_bias)

    def pruning(self):
        """
        Perform pruning
        """

        # prune pre-resnet on cifar
        if self.block_type in ["slim_preresnet", "slim_qpreresnet"]:
            self.pruning_gate_preresnet()
        else:
            assert False, "invalid block type: " + self.block_type


class ResSeqPrune(object):
    """
    Sequantial pruning
    """

    def __init__(self, sequential, current_block_indicator, previsou_block_indicator, seq_type):
        self.current_block_indicator = current_block_indicator
        self.previsou_block_indicator = previsou_block_indicator
        self.sequential = sequential
        self.sequential_length = len(list(self.sequential))
        self.res_block_prune = []

        for i in range(self.sequential_length):
            self.res_block_prune.append(
                ResBlockPrune(self.sequential[i], self.current_block_indicator, self.previsou_block_indicator, block_type=seq_type)
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
        if self.net_type in ["slim_preresnet", "slim_qpreresnet"]:
            filter_weight = self.model.conv.weight
            n, _, _, _ = filter_weight.shape
            group_size = self.model.group_size
            indicator = self.model.assigned_indicator_1.reshape(-1, 1)
            indicator = indicator.expand(n // group_size, group_size).reshape(n)

            # prune and replace conv1
            select_channels = get_select_channels(indicator)
            thin_weight, thin_bias = get_thin_params(self.model.conv, select_channels, 0)
            self.model.conv = replace_layer(self.model.conv, thin_weight, thin_bias)

            res_seq_prune = [
                ResSeqPrune(self.model.layer1, self.model.assigned_indicator_1, self.model.assigned_indicator_1, seq_type=self.net_type),
                ResSeqPrune(self.model.layer2, self.model.assigned_indicator_2, self.model.assigned_indicator_1, seq_type=self.net_type),
                ResSeqPrune(self.model.layer3, self.model.assigned_indicator_3, self.model.assigned_indicator_2, seq_type=self.net_type),
            ]
            for i in range(3):
                res_seq_prune[i].pruning()

            # prune and replace bn and fc
            filter_weight = self.model.fc.weight
            _, n = filter_weight.shape
            group_size = self.model.group_size
            indicator = self.model.assigned_indicator_3.reshape(-1, 1)
            indicator = indicator.expand(n // group_size, group_size).reshape(n)

            select_channels = get_select_channels(indicator)
            thin_weight, thin_bias = get_thin_params(self.model.bn, select_channels, 0)
            self.model.bn = replace_layer(self.model.bn, thin_weight, thin_bias)

            thin_weight, thin_bias = get_thin_params(self.model.fc, select_channels, 1)
            self.model.fc = replace_layer(self.model.fc, thin_weight, thin_bias)

            self.model.layer1 = res_seq_prune[0].sequential
            self.model.layer2 = res_seq_prune[1].sequential
            self.model.layer3 = res_seq_prune[2].sequential