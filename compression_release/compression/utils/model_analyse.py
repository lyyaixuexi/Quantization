import math

import numpy as np
import torch.nn as nn
from prettytable import PrettyTable

from compression.utils.utils import compute_bops, compute_memory_footprint

from . import model_transform as mt

__all__ = ["ModelAnalyse"]


class ModelAnalyse(object):
    def __init__(self, model, logger):
        self.model = mt.list2sequential(model)
        self.logger = logger
        self.bops = []
        self.madds = []
        self.weight_shapes = []
        self.layer_names = []
        self.filter_nums = []
        self.channel_nums = []
        self.bias_shapes = []
        self.output_shapes = []
        self.input_shapes = []
        self.activation_memory_footprint = []
        self.weight_memory_footprint = []
        self.memory_footprint = []

    def _bops_conv_hook(self, layer, x, out):
        if isinstance(out, tuple):
            _, _, h, w = out[0].shape
        else:
            _, _, h, w = out.shape

        if layer.layer_name == "conv":
            bop = compute_bops(
                layer.kernel_size[0], layer.in_channels, layer.out_channels // layer.groups, h, w,
            )
        else:
            bop = compute_bops(
                layer.kernel_size[0], layer.in_channels, layer.out_channels // layer.groups, h, w,
            )

        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.output_shapes.append(list(out.shape))
        self.filter_nums.append(layer.out_channels)
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.bops.append(bop)

    def _memory_footprint_qconv_hook(self, layer, x, out):
        x = x[0]
        n, c, h, w = x.shape

        activation_memory_footprint = compute_memory_footprint(n, c, h, w)
        weight_memory_footprint = compute_memory_footprint(layer.weight.shape[0], layer.weight.shape[1], 
                                                            layer.weight.shape[2], layer.weight.shape[3])
        memory_footprint = activation_memory_footprint + weight_memory_footprint

        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.input_shapes.append(list(x.shape))
        self.output_shapes.append(list(out.shape))
        self.filter_nums.append(layer.out_channels)
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.activation_memory_footprint.append(activation_memory_footprint)
        self.weight_memory_footprint.append(weight_memory_footprint)
        self.memory_footprint.append(memory_footprint)

    def _madds_conv_hook(self, layer, x, out):
        input = x[0]
        batch_size = input.shape[0]
        output_height, output_width = out.shape[2:]

        kernel_height, kernel_width = layer.kernel_size
        in_channels = layer.in_channels
        out_channels = layer.out_channels
        groups = layer.groups

        filters_per_channel = out_channels // groups
        conv_per_position_flops = kernel_height * kernel_width * in_channels * filters_per_channel

        active_elements_count = batch_size * output_height * output_width

        overall_conv_flops = conv_per_position_flops * active_elements_count

        bias_flops = 0
        if layer.bias is not None:
            bias_flops = out_channels * active_elements_count

        overall_flops = overall_conv_flops + bias_flops
        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.output_shapes.append(list(out.shape))
        self.channel_nums.append(in_channels)
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.madds.append(overall_flops)

    def _madds_linear_hook(self, layer, x, out):
        # compute number of multiply-add
        # layer_madds = layer.weight.size(0) * layer.weight.size(1)
        # if layer.bias is not None:
        #     layer_madds += layer.weight.size(0)
        input = x[0]
        batch_size = input.shape[0]
        overall_flops = int(batch_size * input.shape[1] * out.shape[1])

        bias_flops = 0
        if layer.bias is not None:
            bias_flops = out.shape[1]
        overall_flops = overall_flops + bias_flops
        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.channel_nums.append(input.shape[1])
        self.output_shapes.append(list(out.shape))
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.madds.append(overall_flops)

    def _bops_linear_hook(self, layer, x, out):
        bop = compute_bops(1, layer.in_features, layer.out_features, 1, 1,)

        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.output_shapes.append(list(out.shape))
        self.filter_nums.append(out.shape[0])
        if layer.bias is not None:
            self.bias_shapes.append(list(layer.bias.shape))
        else:
            self.bias_shapes.append([0])
        self.bops.append(bop)

    def _memory_footprint_linear_hook(self, layer, x, out):
        x = x[0]
        n, c = x.shape

        activation_memory_footprint = compute_memory_footprint(n, c, 1, 1)
        weight_memory_footprint = compute_memory_footprint(layer.weight.shape[0], layer.weight.shape[1], 
                                                            1, 1)
        memory_footprint = activation_memory_footprint + weight_memory_footprint

        layer_name = layer.layer_name
        self.layer_names.append(layer_name)
        self.weight_shapes.append(list(layer.weight.shape))
        self.input_shapes.append(list(x.shape))
        self.output_shapes.append(list(out.shape))
        self.filter_nums.append(out.shape[0])
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
                hook_list.append(layer.register_forward_hook(self._bops_conv_hook))
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
                ]
            )
        self.logger.info(output)
        repo_str = "|===>Total BOPs: {:f} MBOPs".format(bops_sum / 1e6)
        self.logger.info(repo_str)

        for hook in hook_list:
            hook.remove()

    def params_count(self):
        params_num_list = []

        output = PrettyTable()
        output.field_names = ["Param name", "Shape", "Dim"]

        self.logger.info("------------------------number of parameters------------------------\n")
        for name, param in self.model.named_parameters():
            param_num = param.numel()
            param_shape = [shape for shape in param.shape]
            params_num_list.append(param_num)
            output.add_row([name, param_shape, param_num])
        self.logger.info(output)

        params_num_list = np.array(params_num_list)
        params_num = params_num_list.sum()
        self.logger.info("|===>Number of parameters is: {:}, {:f} M".format(params_num, params_num / 1e6))
        return params_num

    def madds_compute(self, x):
        """
        Compute number of multiply-adds of the model
        """

        hook_list = []
        self.madds = []
        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                hook_list.append(layer.register_forward_hook(self._madds_conv_hook))
                layer.layer_name = layer_name
                # self.layer_names.append(layer_name)
            elif isinstance(layer, nn.Linear):
                hook_list.append(layer.register_forward_hook(self._madds_linear_hook))
                layer.layer_name = layer_name
                # self.layer_names.append(layer_name)
        # run forward for computing FLOPs
        self.model.eval()
        self.model(x)

        madds_np = np.array(self.madds)
        madds_sum = float(madds_np.sum())
        percentage = madds_np / madds_sum

        output = PrettyTable()
        output.field_names = ["Layer", "Weight Shape", "#Channels", "Bias Shape", "Output Shape", "Madds", "Percentage"]

        self.logger.info("------------------------Madds------------------------\n")
        for i in range(len(self.madds)):
            output.add_row([self.layer_names[i], self.weight_shapes[i], self.channel_nums[i], self.bias_shapes[i],
                            self.output_shapes[i], madds_np[i], percentage[i]])
        self.logger.info(output)
        repo_str = "|===>Total MAdds: {:f} M".format(madds_sum / 1e6)
        self.logger.info(repo_str)

        for hook in hook_list:
            hook.remove()

        return madds_np

    def memory_footprint_compute(self, x):
        hook_list = []
        self.bops = []

        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                hook_list.append(layer.register_forward_hook(self._memory_footprint_qconv_hook))
            elif isinstance(layer, nn.Linear):
                hook_list.append(layer.register_forward_hook(self._memory_footprint_linear_hook))
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
                ]
            )
        self.logger.info(output)
        repo_str = "|===>Total Weight Footprint: {:f} KB".format(weight_footprint_sum / 8 / 1e3)
        self.logger.info(repo_str)
        repo_str = "|===>Total Activation Footprint: {:f} KB".format(activation_footprint_sum / 8 / 1e3)
        self.logger.info(repo_str)
        repo_str = "|===>Total Activation Max Footprint: {:f} KB".format(activation_footprint_max / 8 / 1e3)
        self.logger.info(repo_str)
        repo_str = "|===>Total Footprint: {:f} KB".format(total_footprint / 8 / 1e3)
        self.logger.info(repo_str)
        repo_str = "|===>Total Footprint Max: {:f} KB".format(total_footprint_max / 8 / 1e3)
        self.logger.info(repo_str)

        for hook in hook_list:
            hook.remove()
