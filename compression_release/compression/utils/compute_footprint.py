import torch
import math
import torch.nn as nn
from compression.models.resnet import ResNet
from compression.models.quantization.qresnet import QResNet
from compression.models.quantization.qpreresnet import QPreResNet
from compression.models.quantization.qmobilenetv2 import QMobileNetV2
from compression.models.preresnet import PreResNet
from compression.utils.qmodel_analyse import QModelAnalyse
from compression.utils.logger import get_logger

def forward_hook(module, input, output):
    output_shape = output.shape
    module.output_shape = output_shape


def main():
    bits_weight = 32
    bits_activation = 32
    model = QPreResNet(depth=56, num_classes=100, bits_weights=bits_weight, bits_activations=bits_activation)
    # model = QPreResNet(depth=56, num_classes=100, bits_weights=bits_weight, bits_activations=bits_activation, quantize_first_last=True)
    # model = QResNet(18, num_classes=1000, bits_weights=bits_weight, bits_activations=bits_activation, quantize_first_last=True)
    # model = QMobileNetV2(bits_weights=bits_weight, bits_activations=bits_activation)
    logger = get_logger('/tmp/', "baseline")
    logger.info(model)
    model = QModelAnalyse(model, logger)
    random_input = torch.randn(1, 3, 32, 32)
    # random_input = torch.randn(1, 3, 224, 224)
    model.memory_footprint_compute(random_input)

if __name__ == "__main__":
    main()

