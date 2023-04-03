import math
import csv

import numpy as np
import torch.nn as nn
from prettytable import PrettyTable

from compression.models.quantization.dorefa_clip import QConv2d, QLinear
from compression.utils.utils import *

from . import model_transform as mt

__all__ = ["QModelLatencyAnalyse"]


def get_latency(key_name, latency_dict):
    return latency_dict[key_name]


class QModelLatencyAnalyse(object):
    def __init__(self, model, net_type, logger):
        self.model = mt.list2sequential(model)
        self.logger = logger
        self.latencies = []
        self.poweres = []
        self.energies = []
        self.weight_shapes = []
        self.layer_names = []
        self.filter_nums = []
        self.bias_shapes = []
        self.output_shapes = []
        self.bits_weights = []
        self.bits_activations = []
        self.latency_dict = {}
        self.power_dict = {}
        self.net_type = net_type
        # with open('/home/liujing/Models/constraint_time/resnet/resnet-layer-wise-whole.csv') as f:
        with open('/home/liujing/Models/constraint_time/mobilenet/mobilenet-layer-wise-whole.csv') as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            for row in f_csv:
                key_name = row[0]
                latency = row[1]
                power = row[2]
                self.latency_dict[key_name] = latency
                self.power_dict[key_name] = power

    def _latency_qconv_hook(self, layer, x, out):
        input_x = x[0]
        _, _, h, w = input_x.shape

        if layer.layer_name == "conv" or layer.layer_name == "conv1":
            if "mobilenet" in self.net_type:
                latency = 0.29583475
                power = 0.023902358809643077
            elif "resnet" in self.net_type:
                latency = 0.688928125
                power = 0.026348114226714795
            energy = latency * power * 1000 / 16
        else:
            key_name = "width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_SAME-group_{}-fb_{}-wb_{}-layer".format(
                w, h, layer.in_channels, layer.out_channels, layer.kernel_size[0], layer.stride[0], layer.groups, int(layer.bits_weights), int(layer.bits_activations)
            )
            latency = float(self.latency_dict[key_name])
            power = float(self.power_dict[key_name])
            energy = latency * power * 1000 / 16

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
        self.latencies.append(latency)
        self.poweres.append(power)
        self.energies.append(energy)

    def _latency_linear_hook(self, layer, x, out):
        if "mobilenetv1" in self.net_type:
            latency = 0.0062293750000000005
            power = 0.02610436248637335
        elif "mobilenetv2" in self.net_type:
            latency = 0.008362875
            power = 0.025906742446500753
        elif "resnet18" in self.net_type:
            latency = 0.003456125
            power = 0.025792875407300374
        elif "resnet50" in self.net_type:
            latency = 0.012458875000000001
            power = 0.02607727631511384
        energy = latency * power * 1000 / 16

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
        self.latencies.append(latency)
        self.poweres.append(power)
        self.energies.append(energy)

    def latency_compute(self, x):
        hook_list = []
        self.latencies = []

        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                hook_list.append(layer.register_forward_hook(self._latency_qconv_hook))
            elif isinstance(layer, nn.Linear):
                hook_list.append(layer.register_forward_hook(self._latency_linear_hook))
            layer.layer_name = layer_name

        # run forward for computing BOPs
        self.model.eval()
        self.model(x)

        latencies_np = np.array(self.latencies)
        latencies_sum = float(latencies_np.sum())
        # percentage = latencies_np / latencies_sum

        poweres_np = np.array(self.poweres)

        energies_np = np.array(self.energies)
        energies_sum = float(energies_np.sum())

        output = PrettyTable()
        output.field_names = [
            "Layer",
            "Weight Shape",
            "#Filters",
            "Bias Shape",
            "Output Shape",
            "Latency (ms)",
            "Power (mW)",
            "Energy (muJ)",
            "BitW",
            "BitA",
        ]

        self.logger.info("------------------------Latencies------------------------\n")
        for i in range(len(self.latencies)):
            output.add_row(
                [
                    self.layer_names[i],
                    self.weight_shapes[i],
                    self.filter_nums[i],
                    self.bias_shapes[i],
                    self.output_shapes[i],
                    latencies_np[i],
                    poweres_np[i],
                    energies_np[i],
                    self.bits_weights[i],
                    self.bits_activations[i],
                ]
            )
        self.logger.info(output)
        repo_str = "|===>Total Latencies: {:f} ms".format(latencies_sum)
        self.logger.info(repo_str)
        repo_str = "|===>Total energy: {:f} mJ".format(energies_sum)
        self.logger.info(repo_str)

        for hook in hook_list:
            hook.remove()
