import torch
import csv
import torch.nn as nn
import numpy as np
from compression.models.quantization.qmobilenetv2 import QMobileNetV2, QMobileBottleneck
from compression.utils.logger import get_logger

def get_latency_power_dict():
    latency_dict = {}
    power_dict = {}
    # with open('/home/liujing/Models/constraint_time/resnet/layer-wise.csv') as f:
    with open('/home/liujing/Models/mixed_precision/mobilenet-layer-wise-whole.csv') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            key_name = row[0]
            latency = float(row[1])
            power = float(row[2])
            latency_dict[key_name] = latency
            power_dict[key_name] = power
    return latency_dict, power_dict

def conv_forward_hook(module, input, output):
    input_x = input[0]
    n, c, h, w = input_x.shape
    output_shape = output.shape
    module.output_shape = output_shape
    module.h = h
    module.w = w

def fc_forward_hook(module, input, output):
    input_x = input[0]
    _, _, h, w = input_x.shape
    output_shape = output.shape
    module.output_shape = output_shape
    module.h = 1
    module.w = 1

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
        if isinstance(module, (nn.Conv2d)):
            module.register_forward_hook(conv_forward_hook)

    model(random_input)
    bits = [1, 2, 3, 4, 5, 6, 7, 8]
    total_latency = []
    conv_latency = []
    for i in bits:
        w_latency = []
        for j in bits:
            w_latency.append(0.29583475)
        conv_latency.append(w_latency)
    total_latency.append(conv_latency)

    conv_index = 0
    for name, layer in model.named_modules():
        conv_latency = []
        if isinstance(layer, (nn.Conv2d)):
            conv_index += 1
            if conv_index == 1:
                continue
            for i in bits:
                w_latency = []
                for j in bits:
                    key_name = "width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_SAME-group_{}-fb_{}-wb_{}-layer".format(
                            layer.w, layer.h, layer.in_channels, layer.out_channels, layer.kernel_size[0], layer.stride[0], layer.groups, j, i
                        )
                    if key_name in latency_dict:
                        # print(latency_dict[key_name])
                        w_latency.append(latency_dict[key_name])
                        # continue
                    else:
                        # w_latency.append(0.29583475)
                        print("Key name {} not in dict".format(key_name))
                if len(w_latency) != 0:
                    conv_latency.append(w_latency)
            if len(conv_latency) != 0:
                total_latency.append(conv_latency)
    
    fconv_latency = []
    for i in bits:
        w_latency = []
        for j in bits:
            w_latency.append(0.008362875)
        fconv_latency.append(w_latency)
    total_latency.append(fconv_latency)
    total_latency = np.array(total_latency)
    print(total_latency)
    print(total_latency.shape)
    print("Total latency: {}ms".format(np.sum(total_latency[:, 7, 7])))
    np.save('/home/liujing/Models/mixed_precision/qmobilenetv2_imagenet100_batch16_latency_table.npy', total_latency)

if __name__ == "__main__":
    main()

