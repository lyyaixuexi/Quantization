import torch
import math
import csv
import torch.nn as nn
from compression.models.resnet import ResNet
from compression.models.quantization.qresnet import QResNet, QBasicBlock
from compression.models.quantization.qpreresnet import QPreResNet
from compression.models.quantization.qmobilenetv2 import QMobileNetV2, QMobileBottleneck
from compression.models.preresnet import PreResNet
from compression.utils.qmodel_analyse import QModelAnalyse
from compression.utils.logger import get_logger

def get_latency_power_dict():
    latency_dict = {}
    power_dict = {}
    with open('/home/liujing/Models/constraint_time/resnet/resnet-layer-wise-whole.csv') as f:
    # with open('/home/liujing/Models/constraint_time/mobilenet/layer-wise.csv') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            key_name = row[0]
            latency = row[1]
            power = row[2]
            latency_dict[key_name] = latency
            power_dict[key_name] = power
    return latency_dict, power_dict

def forward_hook(module, input, output):
    input_x = input[0]
    n, c, h, w = input_x.shape
    output_shape = output.shape
    module.output_shape = output_shape
    module.h = h
    module.w = w

def main():
    bits_weight = 8
    bits_activation = 8
    model = QResNet(18, num_classes=1000, bits_weights=bits_weight, bits_activations=bits_activation, quantize_first_last=True)
    logger = get_logger('/tmp/', "baseline")
    logger.info(model)
    # random_input = torch.randn(1, 3, 32, 32)
    random_input = torch.randn(1, 3, 224, 224)
    latency_dict, power_dict = get_latency_power_dict()

    for name, module in model.named_modules():
        if isinstance(module, (QBasicBlock)):
            module.conv1.register_forward_hook(forward_hook)
            module.conv2.register_forward_hook(forward_hook)
            if module.downsample:
                module.downsample[0].register_forward_hook(forward_hook)
        elif isinstance(module, (QMobileBottleneck)):
            if module.expand != 1:
                module.conv1.register_forward_hook(forward_hook)
            module.conv2.register_forward_hook(forward_hook)
            module.conv3.register_forward_hook(forward_hook)

    model(random_input)
    bits = [2, 4, 8]
    group_size = 4
    for name, module in model.named_modules():
        if isinstance(module, (QBasicBlock)):
            layer = module.conv1
            num_choices = layer.out_channels // group_size
            print("layer: {}.conv1".format(name))
            for i in bits:
                for j in bits:
                    for k in range(num_choices):
                        key_name = "width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_SAME-group_1-fb_{}-wb_{}-layer".format(
                            layer.w, layer.h, layer.in_channels, (k + 1) * group_size, layer.kernel_size[0], layer.stride[0], i, j
                        )
                        latency = latency_dict[key_name]
                        print('Key name: {}, latency: {}'.format(key_name, latency))
                        if key_name in latency_dict:
                            continue
                        else:
                            print("Key name {} not in dict".format(key_name))
                    # conv1_latency = latency_dict[key_name]

            layer = module.conv2
            print("layer: {}.conv2".format(name))
            for i in bits:
                for j in bits:
                    for k in range(num_choices):
                        key_name = "width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_SAME-group_1-fb_{}-wb_{}-layer".format(
                            layer.w, layer.h, (k + 1) * group_size, layer.out_channels, layer.kernel_size[0], layer.stride[0], i, j
                        )
                        latency = latency_dict[key_name]
                        print('Key name: {}, latency: {}'.format(key_name, latency))
                        if key_name in latency_dict:
                            continue
                        else:
                            print("Key name {} not in dict".format(key_name))

            downsample_latency = 0
            if module.downsample:
                print("layer: {}.downsample".format(name))
                layer = module.downsample[0]
                for i in bits:
                    for j in bits:
                        key_name = "width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_SAME-group_1-fb_{}-wb_{}-layer".format(
                            layer.w, layer.h, layer.in_channels, layer.out_channels, layer.kernel_size[0], layer.stride[0], i, j
                        )
                        latency = latency_dict[key_name]
                        print('Key name: {}, latency: {}'.format(key_name, latency))
                        if key_name in latency_dict:
                            continue
                        else:
                            print("Key name {} not in dict".format(key_name))

if __name__ == "__main__":
    main()

