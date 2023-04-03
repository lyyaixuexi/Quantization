import csv

import torch

import compression.models as md


def get_model(dataset, net_type, depth, n_classes, settings):
    """
    Available model
    cifar:
        preresnet
        vgg
    imagenet:
        resnet
    """

    if dataset in ["m3_imagenet", "imagenet", "imagenet100"]:
        test_input = torch.randn(1, 3, 224, 224)
        if net_type == "resnet":
            model = md.ResNet(depth=depth, num_classes=n_classes)
        elif "qresnet" in net_type:
            model = md.QResNet(
                depth=depth,
                num_classes=n_classes,
                bits_weights=settings.qw,
                bits_activations=settings.qa,
                quantize_first_last=settings.quantize_first_last,
            )
        elif "super_resnet" in net_type:
            model = md.SuperResNet(depth=depth, num_classes=n_classes)
        elif "super_quan_resnet" in net_type:
            model = md.SuperQuanResNet(
                depth=depth,
                num_classes=n_classes,
                bits_weights_list=settings.bits_weights_list,
                bits_activations_list=settings.bits_activations_list,
            )
        elif "super_joint_quan_resnet" in net_type:
            model = md.SuperJointQuanResNet(depth=depth, num_classes=n_classes)
        elif "super_quan_latency_resnet" in net_type:
            latency_dict = {}
            power_dict = {}
            with open(
                "/home/liujing/Models/constraint_time/resnet/resnet-layer-wise-whole.csv"
            ) as f:
                f_csv = csv.reader(f)
                headers = next(f_csv)
                for row in f_csv:
                    key_name = row[0]
                    latency = row[1]
                    power = row[2]
                    latency_dict[key_name] = latency
                    power_dict[key_name] = power
            model = md.SuperQuanLatencyResNet(
                depth=depth,
                num_classes=n_classes,
                latency_dict=latency_dict,
                power_dict=power_dict,
            )
        elif "super_quan_mobilenetv2" in net_type:
            model = md.SuperQuanMobileNetV2(
                num_classes=n_classes, quantize_first_last=settings.quantize_first_last
            )
        elif "super_quan_mobilenetv1" in net_type:
            model = md.SuperQMobileNetV1(
                num_classes=n_classes, quantize_first_last=settings.quantize_first_last
            )
        elif "super_prune_gate_resnet" in net_type:
            model = md.SuperPrunedGateResNet(depth=depth, num_classes=n_classes)
        elif "super_prune_quan_gate_resnet" in net_type:
            model = md.SuperPrunedQuanGateResNet(depth=depth, num_classes=n_classes)
        elif "super_compress_gate_resnet_group" in net_type:
            model = md.SuperCompressedGateResNetGroup(
                depth=depth,
                num_classes=n_classes,
                group_size=settings.group_size,
                bits_weights_list=settings.bits_weights_list,
                bits_activations_list=settings.bits_activations_list,
            )
        elif "super_compress_gate_resnet" in net_type:
            model = md.SuperCompressedGateResNet(
                depth=depth,
                num_classes=n_classes,
                group_size=settings.group_size,
                bits_weights_list=settings.bits_weights_list,
                bits_activations_list=settings.bits_activations_list,
            )
            # model = md.SuperCompressedGateResNet(depth=depth, num_classes=n_classes, num_choices=settings.num_choices)
        elif "super_compress_gate_latency_resnet" in net_type:
            latency_dict = {}
            power_dict = {}
            with open(
                "/home/liujing/Models/constraint_time/resnet/resnet-layer-wise-whole.csv"
            ) as f:
                f_csv = csv.reader(f)
                headers = next(f_csv)
                for row in f_csv:
                    key_name = row[0]
                    latency = row[1]
                    power = row[2]
                    latency_dict[key_name] = latency
                    power_dict[key_name] = power
            model = md.SuperCompressedGateLatencyResNet(
                depth=depth,
                num_classes=n_classes,
                group_size=settings.group_size,
                latency_dict=latency_dict,
                power_dict=power_dict,
            )
        elif "super_compress_gate_latency_mobilenetv2" in net_type:
            latency_dict = {}
            power_dict = {}
            with open(
                "/home/liujing/Models/constraint_time/mobilenet/mobilenet-layer-wise-whole.csv"
            ) as f:
                f_csv = csv.reader(f)
                headers = next(f_csv)
                for row in f_csv:
                    key_name = row[0]
                    latency = row[1]
                    power = row[2]
                    latency_dict[key_name] = latency
                    power_dict[key_name] = power
            model = md.SuperCompressedLatencyMobileNetV2(
                num_classes=n_classes,
                group_size=settings.group_size,
                latency_dict=latency_dict,
                power_dict=power_dict,
            )
        elif "super_compress_gate_mobilenetv2" in net_type:
            model = md.SuperCompressedMobileNetV2(
                quantize_first_last=settings.quantize_first_last,
                group_size=settings.group_size,
            )
        elif "super_compress_joint_gate_resnet" in net_type:
            model = md.SuperCompressedJointGateResNet(
                depth=depth, num_classes=n_classes, num_choices=settings.num_choices
            )
        elif "nonuniform_resnet" in net_type:
            model = md.NonUniformQResNet(
                depth=depth,
                num_classes=n_classes,
                bits_weights=settings.qw,
                bits_activations=settings.qa,
                T=settings.T,
            )
        elif "mobilenetv1" in net_type:
            if settings.qw == 32 and settings.qa == 32:
                model = md.MobileNetV1()
            else:
                model = md.QMobileNetV1(
                    bits_weights=settings.qw,
                    bits_activations=settings.qa,
                    quantize_first_last=settings.quantize_first_last,
                )
        elif "super_mobilenetv2" in net_type:
            model = md.get_super_mobilenetv2(num_classes=n_classes)
        elif "qmobilenetv2" in net_type:
            model = md.QMobileNetV2(
                bits_weights=settings.qw,
                bits_activations=settings.qa,
                quantize_first_last=settings.quantize_first_last,
            )
        elif "dq_mobilenetv2" in net_type:
            model = md.DQMobileNetV2(quantize_first_last=settings.quantize_first_last)
        elif "dq_resnet" in net_type:
            model = md.DQResNet(
                depth=depth,
                num_classes=n_classes,
                quantize_first_last=settings.quantize_first_last,
            )
        elif "mobilenetv2" in net_type:
            model = md.MobileNetV2()
        elif net_type == "mobilenetv3":
            model = md.MobileNetV3(dropout_rate=0.2)
        elif net_type == "mobilenetv3_small":
            model = md.MobileNetV3Small(dropout_rate=0.2)
        else:
            assert False, "use {} data while network is {}".format(dataset, net_type)
    elif dataset in ["cifar10", "cifar100"]:
        test_input = torch.randn(1, 3, 32, 32)
        if net_type == "preresnet":
            model = md.PreResNet(depth=depth, num_classes=n_classes)
        elif net_type == "slim_preresnet":
            model = md.PreResNet(depth=depth, num_classes=n_classes)
        elif net_type == "qpreresnet":
            model = md.QPreResNet(
                depth=depth,
                num_classes=n_classes,
                bits_weights=settings.qw,
                bits_activations=settings.qa,
                quantize_first_last=settings.quantize_first_last,
            )
        elif net_type == "dorefa_preresnet":
            model = md.DoReFaPreResNet(
                depth=depth,
                num_classes=n_classes,
                bits_weights=settings.qw,
                bits_activations=settings.qa,
                quantize_first_last=settings.quantize_first_last,
            )
        elif net_type == "super_preresnet":
            model = md.SuperPreResNet(
                depth=depth,
                quantize_first_last=settings.quantize_first_last,
                num_classes=n_classes,
            )
        elif net_type == "super_joint_quan_preresnet":
            model = md.SuperJointQuanPreResNet(depth=depth, num_classes=n_classes)
        elif net_type == "super_quan_preresnet":
            model = md.SuperQuanPreResNet(
                depth=depth,
                num_classes=n_classes,
                bits_weights_list=settings.bits_weights_list,
                bits_activations_list=settings.bits_activations_list,
            )
        elif net_type == "super_quan_preresnet_memory":
            model = md.SuperQuanPreResNetMemory(depth=depth, num_classes=n_classes)
        elif "differentiable_quan_preresnet_memory" in net_type:
            model = md.DQPreResNetMemory(depth=depth, num_classes=n_classes)
        elif "differentiable_quan_preresnet" in net_type:
            model = md.DQPreResNet(depth=depth, num_classes=n_classes)
        elif "dnas_preresnet_memory" in net_type:
            model = md.DNASPreResNetMemory(depth=depth, num_classes=n_classes)
        elif "dnas_preresnet" in net_type:
            model = md.DNASPreResNet(depth=depth, num_classes=n_classes)
        elif "dnas_preresnet_clip" in net_type:
            model = md.DNASPreResNetClip(depth=depth, num_classes=n_classes)
        elif "dnas_mobilenetv3_cifar" in net_type:
            model = md.DNASAsyMobileNetV3Cifar(num_classes=n_classes)
        elif net_type == "super_quan_gumbel_preresnet":
            model = md.SuperQuanGumbelPreResNet(depth=depth, num_classes=n_classes)
        # elif net_type == "super_prune_gate_preresnet":
        #     model = md.SuperPrunedGatePreResNet(depth=depth, num_classes=n_classes)
        elif net_type == "super_prune_gate_preresnet":
            model = md.SuperPrunedGatePreResNet(
                depth=depth, num_classes=n_classes, group_size=settings.group_size
            )
        elif net_type == "super_prune_gate_qpreresnet":
            model = md.SuperPrunedGateQPreResNet(
                depth=depth, num_classes=n_classes, group_size=settings.group_size
            )
        elif net_type == "super_prune_gate_memory_preresnet":
            model = md.SuperPrunedGateMemoryPreResNet(
                depth=depth, num_classes=n_classes, group_size=settings.group_size
            )
        elif net_type == "super_slim_gate_preresnet":
            model = md.SuperSlimGatePreResNet(depth=depth, num_classes=n_classes)
        elif net_type == "super_prune_quan_gate_preresnet":
            model = md.SuperPrunedQuanGatePreResNet(depth=depth, num_classes=n_classes)
        elif net_type == "super_compress_gate_preresnet":
            model = md.SuperCompressedGatePreResNet(
                depth=depth,
                num_classes=n_classes,
                group_size=settings.group_size,
                bits_weights_list=settings.bits_weights_list,
                bits_activations_list=settings.bits_activations_list,
            )
        elif net_type == "super_compress_gate_memory_preresnet":
            model = md.SuperCompressedGatePreResNetMemory(
                depth=depth, num_classes=n_classes, group_size=settings.group_size
            )
        elif net_type == "super_compress_joint_gate_preresnet":
            model = md.SuperCompressedJointGatePreResNet(
                depth=depth, num_classes=n_classes, group_size=settings.group_size
            )
        elif net_type == "super_prune_preresnet":
            model = md.SuperPrunedPreResNet(
                depth=depth,
                quantize_first_last=settings.quantize_first_last,
                num_classes=n_classes,
            )
        elif net_type == "super_prune_not_shared_preresnet":
            model = md.SuperPrunedNotShaerdPreResNet(depth=depth, num_classes=n_classes)
        elif net_type == "super_prune_scale_preresnet":
            model = md.SuperPrunedScalePreResNet(depth=depth, num_classes=n_classes)
        elif net_type == "mobilenetv1_cifar":
            if settings.qw == 32 and settings.qa == 32:
                model = md.MobileNetv1CIFAR(num_classes=n_classes)
        elif net_type == "mobilenetv2_cifar":
            if settings.qw == 32 and settings.qa == 32:
                model = md.MobileNetV2Cifar(num_classes=n_classes)
            else:
                model = md.QMobileNetV2Cifar(
                    num_classes=n_classes,
                    bits_weights=settings.qw,
                    bits_activations=settings.qa,
                )
        elif net_type == "mobilenetv2_cifar2":
            if settings.qw == 32 and settings.qa == 32:
                model = md.MobileNetV2CIFAR2(num_classes=n_classes)
            else:
                model = md.QMobileNetV2CIFAR2(
                    num_classes=n_classes,
                    bits_weights=settings.qw,
                    bits_activations=settings.qa,
                )
        elif net_type == "mobilenetv3_cifar":
            if settings.qw == 32 and settings.qa == 32:
                model = md.MobileNetV3Cifar(num_classes=n_classes)
            else:
                model = md.QMobileNetV3Cifar(
                    num_classes=n_classes,
                    bits_weights=settings.qw,
                    bits_activations=settings.qa,
                    quantize_first_last=settings.quantize_first_last,
                )
        elif net_type == "qasymobilenetv3_cifar":
            model = md.QAsyMobileNetV3Cifar(
                num_classes=n_classes,
                bits_weights=settings.qw,
                bits_activations=settings.qa,
                quantize_first_last=settings.quantize_first_last,
            )
        elif net_type == "super_qsy_mobilenetv3_cifar":
            model = md.SuperQAsyMobileNetV3Cifar(
                num_classes=n_classes,
                bits_weights=settings.qw,
                bits_activations=settings.qa,
                quantize_first_last=settings.quantize_first_last,
            )
        elif net_type == "super_prune_gate_mobilenetv3_cifar":
            model = md.SuperPruneAsyMobileNetV3Cifar(
                num_classes=n_classes, group_size=settings.group_size
            )
        elif net_type == "super_compress_qsy_mobilenetv3_cifar":
            model = md.SuperCompressAsyMobileNetV3Cifar(
                num_classes=n_classes,
                bits_weights=settings.qw,
                bits_activations=settings.qa,
                quantize_first_last=settings.quantize_first_last,
                group_size=settings.group_size,
            )
        elif "dq_mobilenetv3_cifar" in net_type:
            model = md.DQAsyMobileNetV3Cifar(
                num_classes=n_classes, quantize_first_last=settings.quantize_first_last
            )
        elif "meta_preresnet" in net_type:
            model = md.MetaQPreResNet(depth=depth, num_classes=n_classes)
        elif "nonuniform_preresnet" in net_type:
            model = md.NonUniformQPreResNet(
                depth=depth,
                num_classes=n_classes,
                bits_weights=settings.qw,
                bits_activations=settings.qa,
                T=settings.T,
            )
        elif "dq_mobilenetv2_cifar" in net_type:
            model = md.DQMobileNetV2Cifar(
                num_classes=n_classes, quantize_first_last=settings.quantize_first_last
            )
        else:
            assert False, "use {} data while network is {}".format(dataset, net_type)
    else:
        assert False, "unsupported data set: {}".format(dataset)
    return model, test_input
