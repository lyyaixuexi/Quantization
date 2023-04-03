import torch
import math
import torch.nn as nn
from compression.models.resnet import ResNet
from compression.models.quantization.qpreresnet import QPreResNet
from compression.models.mobilenetv3 import MobileNetV3
from compression.models.mobilenetv3_small import MobileNetV3Small
# from compression.models.mobilenetv3_han import MobileNetV3
from compression.models.mobilenetv2 import MobileNetV2
from compression.models.mobilenetv3_cifar import MobileNetV3Cifar
from compression.models.preresnet import PreResNet
from compression.utils.model_analyse import ModelAnalyse
from compression.utils.logger import get_logger

def forward_hook(module, input, output):
    output_shape = output.shape
    module.output_shape = output_shape


def main():
    # model = MobileNetV3()
    # model = MobileNetV3Small()
    # model = MobileNetV2()
    bits_weight = 32
    bits_activation = 32
    # model = QPreResNet(depth=20, num_classes=100, bits_weights=bits_weight, bits_activations=bits_activation)
    model = MobileNetV3Cifar(num_classes=100)
    logger = get_logger('/tmp/', "baseline")
    logger.info(model)
    analysis = ModelAnalyse(model, logger)
    random_input = torch.randn(1, 3, 32, 32)
    analysis.madds_compute(random_input)
    # analysis.params_count()

if __name__ == "__main__":
    main()

