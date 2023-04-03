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
    # with open('/home/liujing/Models/constraint_time/resnet/layer-wise.csv') as f:
    with open('/home/liujing/Models/constraint_time/mobilenet/mobilenet-layer-wise-whole.csv') as f:
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
    # model = QResNet(18, num_classes=1000, bits_weights=bits_weight, bits_activations=bits_activation, quantize_first_last=True)
    model = QMobileNetV2(bits_weights=bits_weight, bits_activations=bits_activation, quantize_first_last=True)
    logger = get_logger('/tmp/', "baseline")
    logger.info(model)
    # random_input = torch.randn(1, 3, 32, 32)
    random_input = torch.randn(1, 3, 224, 224)
    latency_dict, power_dict = get_latency_power_dict()

    for name, module in model.named_modules():
        if isinstance(module, (QMobileBottleneck)):
            if module.expand != 1:
                module.conv1.register_forward_hook(forward_hook)
            module.conv2.register_forward_hook(forward_hook)
            module.conv3.register_forward_hook(forward_hook)

    model(random_input)
    bits = [2, 4, 8]
    group_size = 4
    for name, module in model.named_modules():
        conv1_latency = 0
        if isinstance(module, (QMobileBottleneck)):
            step = max(module.conv2.out_channels // 64, 4)
            n_chocies = module.conv2.out_channels // step
            if module.expand != 1:
                layer = module.conv1
                print("layer: {}.conv1, step: {}".format(name, step))
                for i in bits:
                    for j in bits:
                        for k in range(n_chocies):
                            key_name = "width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_SAME-group_{}-fb_{}-wb_{}-layer".format(
                                layer.w, layer.h, layer.in_channels, (k + 1) * step, layer.kernel_size[0], layer.stride[0], layer.groups, i, j
                            )
                            if key_name in latency_dict:
                                print(key_name)
                                continue
                            else:
                                print("Key name {} not in dict".format(key_name))

            layer = module.conv2
            print("layer: {}.conv2".format(name))
            for i in bits:
                for j in bits:
                    for k in range(n_chocies):
                        key_name = "width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_SAME-group_{}-fb_{}-wb_{}-layer".format(
                            layer.w, layer.h, (k + 1) * step, (k + 1) * step, layer.kernel_size[0], layer.stride[0], (k + 1) * step, i, j
                        )
                        if key_name in latency_dict:
                            print(key_name)
                            continue
                        else:
                            print("Key name {} not in dict".format(key_name))

            layer = module.conv3
            print("layer: {}.conv3".format(name))
            for i in bits:
                for j in bits:
                    for k in range(n_chocies):
                        key_name = "width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_SAME-group_{}-fb_{}-wb_{}-layer".format(
                            layer.w, layer.h, (k + 1) * step, layer.out_channels, layer.kernel_size[0], layer.stride[0], layer.groups, i, j
                        )
                        if key_name in latency_dict:
                            print(key_name)
                            continue
                        else:
                            print("Key name {} not in dict".format(key_name))

            assert False

if __name__ == "__main__":
    main()

