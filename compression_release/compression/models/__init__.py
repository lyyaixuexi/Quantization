from .mobilenetv1 import *
from .mobilenetv1_cifar import MobileNetv1CIFAR
from .mobilenetv2 import *
from .mobilenetv2_cifar import MobileNetV2Cifar
from .mobilenetv2_cifar2 import MobileNetV2CIFAR2
from .mobilenetv3 import MobileNetV3
from .mobilenetv3_cifar import MobileNetV3Cifar
from .mobilenetv3_small import MobileNetV3Small
from .preresnet import *
from .quantization.dnas_memory_preresnet import DNASPreResNetMemory
from .quantization.dnas_preresnet import DNASPreResNet
from .quantization.dnas_preresnet_clip import DNASPreResNetClip
from .quantization.dnas_qasy_mobilenetv3_cifar import DNASAsyMobileNetV3Cifar
from .quantization.dorefa_preresnet import DoReFaPreResNet
from .quantization.dq_memory_preresnet import DQPreResNetMemory
from .quantization.dq_mobilenetv2_cifar import DQMobileNetV2Cifar
from .quantization.dq_mobilenetv3_cifar import DQAsyMobileNetV3Cifar
from .quantization.dq_preresnet import DQPreResNet
from .quantization.dq_qmobilenetv2 import DQMobileNetV2
from .quantization.dq_qresnet import DQResNet
from .quantization.meta_qpreresnet import MetaQPreResNet
from .quantization.non_uniform_qresnet import NonUniformQResNet
from .quantization.non_unifrom_qpreresnet import NonUniformQPreResNet
from .quantization.qasy_mobilenetv3_cifar import QAsyMobileNetV3Cifar
from .quantization.qmobilenetv1 import QMobileNetV1
from .quantization.qmobilenetv2 import QMobileNetV2
from .quantization.qmobilenetv2_cifar import QMobileNetV2Cifar
from .quantization.qmobilenetv2_cifar2 import QMobileNetV2CIFAR2
from .quantization.qmobilenetv3_cifar import QMobileNetV3Cifar
from .quantization.qpreresnet import QPreResNet
from .quantization.qresnet import QResNet
from .quantization.super_compress_gate_latency_mobilenetv2 import (
    SuperCompressedLatencyMobileNetV2,
)
from .quantization.super_compress_gate_latency_resnet import (
    SuperCompressedGateLatencyResNet,
)
from .quantization.super_compress_gate_memory_preresnet import (
    SuperCompressedGatePreResNetMemory,
)
from .quantization.super_compress_gate_mobilenetv2 import SuperCompressedMobileNetV2
from .quantization.super_compress_gate_mobilenetv3_cifar import (
    SuperCompressAsyMobileNetV3Cifar,
)
from .quantization.super_compress_gate_preresnet import SuperCompressedGatePreResNet
from .quantization.super_compress_gate_resnet import SuperCompressedGateResNet
from .quantization.super_compress_gate_resnet_group import (
    SuperCompressedGateResNetGroup,
)
from .quantization.super_compress_joint_gate_preresnet import (
    SuperCompressedJointGatePreResNet,
)
from .quantization.super_compress_joint_gate_resnet import (
    SuperCompressedJointGateResNet,
)
from .quantization.super_joint_quan_preresnet import SuperJointQuanPreResNet
from .quantization.super_joint_quan_resnet import SuperJointQuanResNet
from .quantization.super_preresnet import SuperPreResNet
from .quantization.super_prune_gate_memory_preresnet import (
    SuperPrunedGateMemoryPreBasicBlock,
    SuperPrunedGateMemoryPreResNet,
)
from .quantization.super_prune_gate_mobilenetv3_cifar import (
    SuperPruneAsyMobileNetV3Cifar,
)
from .quantization.super_prune_gate_preresnet import (
    SuperPrunedGatePreBasicBlock,
    SuperPrunedGatePreResNet,
)
from .quantization.super_prune_gate_qpreresnet import SuperPrunedGateQPreResNet
from .quantization.super_prune_gate_resnet import (
    SuperPrunedGateBasicBlock,
    SuperPrunedGateBottleneck,
    SuperPrunedGateResNet,
)

# from .quantization.super_prune_gate_preresnet import SuperPrunedGatePreResNet
# from .quantization.super_prune_preresnet import (
#     SuperPrunedPreBasicBlock,
#     SuperPrunedPreResNet,
# )
# from .quantization.super_prune_scale_preresnet import (
#     SuperPrunedScalePreBasicBlock,
#     SuperPrunedScalePreResNet,
# )
from .quantization.super_prune_quan_gate_preresnet import (
    SuperPrunedQuanGatePreBasicBlock,
    SuperPrunedQuanGatePreResNet,
)
from .quantization.super_prune_quan_gate_resnet import SuperPrunedQuanGateResNet
from .quantization.super_qasy_mobilenetv3_cifar import SuperQAsyMobileNetV3Cifar
from .quantization.super_quan_gumbel_preresnet import SuperQuanGumbelPreResNet
from .quantization.super_quan_memory_preresnet import SuperQuanPreResNetMemory
from .quantization.super_quan_mobilenetv1 import SuperQuanMobileNetV1
from .quantization.super_quan_mobilenetv2 import SuperQuanMobileNetV2
from .quantization.super_quan_preresnet import SuperQuanPreResNet
from .quantization.super_quan_resnet import SuperQuanResNet
from .quantization.super_quan_resnet_latency import SuperQuanLatencyResNet
from .quantization.super_resnet import SuperResNet
from .quantization.super_slim_gate_preresnet import SuperSlimGatePreResNet
from .resnet import *
from .super_mobilenetv2 import get_super_mobilenetv2
from .vgg import *

# from .quantization.super_prune_not_shared_preresnet import (
#     SuperPrunedNotShaerdPreResNet,
#     SuperPrunedNotSharedPreBasicBlock,
# )
