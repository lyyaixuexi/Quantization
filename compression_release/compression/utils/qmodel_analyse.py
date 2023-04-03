import math

import numpy as np
import torch.nn as nn
from compression.models.quantization.dorefa_clip import QConv2d, QLinear
from compression.utils.utils import *
from prettytable import PrettyTable

from . import model_transform as mt

__all__ = ["QModelAnalyse"]


class QModelAnalyse(object):
    def __init__(self, model, logger):
        self.model = mt.list2sequential(model)
        self.logger = logger
        self.weight_memory_footprint = []
        self.activation_memory_footprint = []
        self.memory_footprint = []
        self.bops = []
        self.weight_shapes = []
        self.layer_names = []
        self.filter_nums = []
        self.bias_shapes = []
        self.input_shapes = []
        self.output_shapes = []
        self.bits_weights = []
        self.bits_activations = []

    def _bops_qconv_hook(self, layer, x, out):
        if isinstance(out, tuple):
            _, _, h, w = out[0].shape
        else:
            _, _, h, w = out.shape

        if layer.layer_name == "conv" or layer.layer_name == "conv1":
            if hasattr(layer, "bits_weights"):
                # bitw = 8
                bita = 8
                bop = compute_bops(
                    layer.kernel_size[0],
                    layer.in_channels,
                    layer.out_channels // layer.groups,
                    h,
                    w,
                    layer.bits_weights,
                    bita,
                )
                # layer.bits_weights = bitw
                # layer.bits_activations = bita
            else:
                bop = compute_bops(
                    layer.kernel_size[0],
                    layer.in_channels,
                    layer.out_channels // layer.groups,
                    h,
                    w,
                    32,
                    32,
                )
                layer.bits_weights = 32
                layer.bits_activations = 32
            # bop = 0
        else:
            bop = compute_bops(
                layer.kernel_size[0],
                layer.in_channels,
                layer.out_channels // layer.groups,
                h,
                w,
                layer.bits_weights,
                layer.bits_activations,
            )

        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.output_shapes.append(list(out.shape))
        self.filter_nums.append(layer.out_channels)
        self.bits_weights.append(layer.bits_weights)
        self.bits_activations.append(layer.bits_activations)
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.bops.append(bop)

    def _memory_footprint_qconv_hook(self, layer, x, out):
        x = x[0]
        n, c, h, w = x.shape

        if layer.layer_name == "conv" or layer.layer_name == "conv1":
            if hasattr(layer, "bits_weights"):
                # bitw = 8
                bita = 8
                activation_memory_footprint = compute_memory_footprint(n, c, h, w, bita)
                weight_memory_footprint = compute_memory_footprint(
                    layer.weight.shape[0],
                    layer.weight.shape[1],
                    layer.weight.shape[2],
                    layer.weight.shape[3],
                    layer.bits_weights,
                )
                memory_footprint = activation_memory_footprint + weight_memory_footprint
            else:
                activation_memory_footprint = compute_memory_footprint(n, c, h, w, 32)
                weight_memory_footprint = compute_memory_footprint(
                    layer.weight.shape[0],
                    layer.weight.shape[1],
                    layer.weight.shape[2],
                    layer.weight.shape[3],
                    32,
                )
                memory_footprint = activation_memory_footprint + weight_memory_footprint
                layer.bits_weights = 32
                layer.bits_activations = 32
            # bop = 0
        else:
            activation_memory_footprint = compute_memory_footprint(
                n, c, h, w, layer.bits_activations
            )
            weight_memory_footprint = compute_memory_footprint(
                layer.weight.shape[0],
                layer.weight.shape[1],
                layer.weight.shape[2],
                layer.weight.shape[3],
                layer.bits_weights,
            )
            memory_footprint = activation_memory_footprint + weight_memory_footprint

        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.input_shapes.append(list(x.shape))
        self.output_shapes.append(list(out.shape))
        self.filter_nums.append(layer.out_channels)
        self.bits_weights.append(layer.bits_weights)
        self.bits_activations.append(layer.bits_activations)
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.activation_memory_footprint.append(activation_memory_footprint)
        self.weight_memory_footprint.append(weight_memory_footprint)
        self.memory_footprint.append(memory_footprint)

    def _bops_linear_hook(self, layer, x, out):
        if hasattr(layer, "bits_weights"):
            # bitw = 8
            # bita = 8
            bop = compute_bops(
                1,
                layer.in_features,
                layer.out_features,
                1,
                1,
                layer.bits_weights,
                layer.bits_activations,
            )
            # layer.bits_weights = bitw
            # layer.bits_activations = bita
        else:
            bop = compute_bops(1, layer.in_features, layer.out_features, 1, 1, 32, 32)
            layer.bits_weights = 32
            layer.bits_activations = 32
        # bop = 0

        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.output_shapes.append(list(out.shape))
        self.filter_nums.append(out.shape[0])
        self.bits_weights.append(layer.bits_weights)
        self.bits_activations.append(layer.bits_activations)
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.bops.append(bop)

    def _memory_footprint_linear_hook(self, layer, x, out):
        x = x[0]
        n, c = x.shape
        if hasattr(layer, "bits_weights"):
            activation_memory_footprint = compute_memory_footprint(
                n, c, 1, 1, layer.bits_activations
            )
            weight_memory_footprint = compute_memory_footprint(
                layer.weight.shape[0], layer.weight.shape[1], 1, 1, layer.bits_weights
            )
            memory_footprint = activation_memory_footprint + weight_memory_footprint
        else:
            activation_memory_footprint = compute_memory_footprint(n, c, 1, 1)
            weight_memory_footprint = compute_memory_footprint(
                layer.weight.shape[0], layer.weight.shape[1], 1, 1
            )
            memory_footprint = activation_memory_footprint + weight_memory_footprint
            layer.bits_weights = 32
            layer.bits_activations = 32
        # bop = 0

        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.input_shapes.append(list(x.shape))
        self.output_shapes.append(list(out.shape))
        self.filter_nums.append(out.shape[0])
        self.bits_weights.append(layer.bits_weights)
        self.bits_activations.append(layer.bits_activations)
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.activation_memory_footprint.append(activation_memory_footprint)
        self.weight_memory_footprint.append(weight_memory_footprint)
        self.memory_footprint.append(memory_footprint)

    def bops_compute(self, x):
        hook_list = []
        self.bops = []

        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                hook_list.append(layer.register_forward_hook(self._bops_qconv_hook))
            elif isinstance(layer, nn.Linear):
                hook_list.append(layer.register_forward_hook(self._bops_linear_hook))
            layer.layer_name = layer_name

        # run forward for computing BOPs
        self.model.eval()
        self.model(x)

        bops_np = np.array(self.bops)
        bops_sum = float(bops_np.sum())
        percentage = bops_np / bops_sum

        output = PrettyTable()
        output.field_names = [
            "Layer",
            "Weight Shape",
            "#Filters",
            "Bias Shape",
            "Output Shape",
            "BOPs",
            "Percentage",
            "BitW",
            "BitA",
        ]

        self.logger.info("------------------------BOPs------------------------\n")
        for i in range(len(self.bops)):
            output.add_row(
                [
                    self.layer_names[i],
                    self.weight_shapes[i],
                    self.filter_nums[i],
                    self.bias_shapes[i],
                    self.output_shapes[i],
                    bops_np[i],
                    percentage[i],
                    self.bits_weights[i],
                    self.bits_activations[i],
                ]
            )
        self.logger.info(output)
        repo_str = "|===>Total BOPs: {:f} MBOPs".format(bops_sum / 1e6)
        self.logger.info(repo_str)

        for hook in hook_list:
            hook.remove()

    def memory_footprint_compute(self, x):
        hook_list = []
        self.bops = []

        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                hook_list.append(
                    layer.register_forward_hook(self._memory_footprint_qconv_hook)
                )
            elif isinstance(layer, nn.Linear):
                hook_list.append(
                    layer.register_forward_hook(self._memory_footprint_linear_hook)
                )
            layer.layer_name = layer_name

        # run forward for computing BOPs
        self.model.eval()
        self.model(x)

        weight_footprint = np.array(self.weight_memory_footprint)
        weight_footprint_sum = float(weight_footprint.sum())
        activation_footprint = np.array(self.activation_memory_footprint)
        activation_footprint_sum = float(activation_footprint.sum())
        activation_footprint_max = float(activation_footprint.max())
        total_footprint = weight_footprint_sum + activation_footprint_sum
        total_footprint_max = weight_footprint_sum + activation_footprint_max

        output = PrettyTable()
        output.field_names = [
            "Layer",
            "Weight Shape",
            "#Filters",
            "Bias Shape",
            "Input Shape",
            "Weight Footprint",
            "Activation Footprint",
            "BitW",
            "BitA",
        ]

        self.logger.info("------------------------BOPs------------------------\n")
        for i in range(len(self.weight_memory_footprint)):
            output.add_row(
                [
                    self.layer_names[i],
                    self.weight_shapes[i],
                    self.filter_nums[i],
                    self.bias_shapes[i],
                    self.input_shapes[i],
                    weight_footprint[i],
                    activation_footprint[i],
                    self.bits_weights[i],
                    self.bits_activations[i],
                ]
            )
        self.logger.info(output)
        repo_str = "|===>Total Weight Footprint: {:f} KB".format(
            weight_footprint_sum / 8 / 1e3
        )
        self.logger.info(repo_str)
        repo_str = "|===>Total Activation Footprint: {:f} KB".format(
            activation_footprint_sum / 8 / 1e3
        )
        self.logger.info(repo_str)
        repo_str = "|===>Total Activation Max Footprint: {:f} KB".format(
            activation_footprint_max / 8 / 1e3
        )
        self.logger.info(repo_str)
        repo_str = "|===>Total Footprint: {:f} KB".format(total_footprint / 8 / 1e3)
        self.logger.info(repo_str)
        repo_str = "|===>Total Footprint Max: {:f} KB".format(
            total_footprint_max / 8 / 1e3
        )
        self.logger.info(repo_str)

        for hook in hook_list:
            hook.remove()
