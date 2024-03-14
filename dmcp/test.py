import argparse
import utils.distributed as dist
import utils.tools as tools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torch.distributed as dist
import torch.multiprocessing as mp

import torch.backends.cudnn as cudnn
import tflite_quantization_PACT_weight_and_act as tflite
from overflow_utils import *

from models.adaptive.resnet import AdaptiveBasicBlock


def setDDP(model, args):
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    return model


def convert_to_full_int(net):
    # inference one time to update M and n
    net.eval()

    # tflite.update_next_act_scale(model)
    # 　replace float param to interger param
    index = 0
    act_scale_list = []
    for name, layer in net.named_modules():
        if isinstance(layer, (tflite.Conv2d_quantization)):
            act_scale_list.append(layer.act_scale)
            ######################################
            layer.inference_type = "full_int"
            ######################################

            layer.index = index
            index += 1
            # print some information
            print(
                "convert_to_full_int act_bits:{}  weight_bits:{} bias_bits:{}".format(layer.act_bits, layer.weight_bits,
                                                                                      layer.bias_bits))

    input_scale = tflite.layer_transform("full_int", net)  ###
    print('input_scale={}'.format(input_scale))

    return act_scale_list, input_scale


def evaluate(runner, loaders):
    train_loader, val_loader = loaders
    runner.infer(val_loader, train_loader=train_loader)


if __name__ == '__main__':
    # TORCH_DISTRIBUTED_DEBUG=DETAIL
    parser = argparse.ArgumentParser(description='DMCP Implementation')
    parser.add_argument('-C', '--config', required=True)
    parser.add_argument('-M', '--mode', default='eval')
    parser.add_argument('-F', '--flops', required=True)
    parser.add_argument('-D', '--data', required=True)
    parser.add_argument('--chcfg', default=None)
    parser.add_argument('--model_path', type=str, default="", help="weight file path")  # weight/darknet53_448.weights
    parser.add_argument('--raw', action='store_true', default=False)

    parser.add_argument('--distributed', action='store_true', default=True,
                        help='disables CUDA training')  ###########
    parser.add_argument('--gpu', default=None, type=str)  ###############

    args = parser.parse_args()

    opt = args
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:63634',
                            world_size=1, rank=0)

    # args = tools.get_args(parser)
    config = tools.get_config(args)

    tools.init(config)
    tb_logger, logger = tools.get_logger(config)

    # tools.check_dist_init(config, logger)
    checkpoint = tools.get_checkpoint(config)
    runner = tools.get_model(args, config, checkpoint)
    loaders = tools.get_data_loader(config)

    if args.raw:
        # if args.mode == 'evaluate':
        evaluate(runner, loaders)
        # elif args.mode == 'calc_flops':
        flops = tools.get_model_flops(config, runner.get_model())
        print('flops: {}'.format(flops))
        # elif args.mode == 'calc_params':
        params = tools.get_model_parameters(runner.get_model())
        print('params: {}'.format(params))
    else:
        model = runner.model
        model = model.module
        if 1:  # opt.quantization:
            model = model.cpu()
            model = tflite.replace(model=model, inference_type="full_int", Mn_aware=True, fuseBN=True)

        if 1:  # opt.Mn_aware:
            tflite.update_next_act_scale(model)

        # model=setDDP(model,args)

        if 1:  # opt.quantization and opt.fuseBN is not None:
            print('Fusing BN and Conv2d_quantization layers... ')
            count = 0
            for name, m in model.named_modules():  # model.module.named_modules():
                print(name)
                if type(m) in [AdaptiveBasicBlock]:
                    m.conv1 = tflite.fuse_conv_and_bn(m.conv1, m.bn1)  # update conv
                    m.conv1.Mn_aware = True
                    delattr(m, 'bn1')  # remove batchnorm

                    m.conv2 = tflite.fuse_conv_and_bn(m.conv2, m.bn2)  # update conv
                    m.conv2.Mn_aware = True
                    delattr(m, 'bn2')  # remove batchnorm

                    if m.downsample is not None:
                        conv = tflite.fuse_conv_and_bn(m.downsample[0], m.downsample[1])  # update conv
                        conv.Mn_aware = True
                        m.downsample = nn.Sequential(conv)

                    m.forward = m.fuseforward  # update forward####
                    if hasattr(m.conv1, 'Mn_aware') and m.conv1.Mn_aware:
                        m.register_buffer('block_M', torch.ones(1))
                        m.register_buffer('block_n', torch.zeros(1))
                        print('have block_M..............................')
                    count += 1
            model.conv1 = tflite.fuse_conv_and_bn(model.conv1, model.bn1)  # update conv
            model.conv1.Mn_aware = True
            delattr(model, 'bn1')  # remove batchnorm
            model.forward = model.fuseforward  #####dDDP
        if opt.model_path is not None:
            state_dict = torch.load(opt.model_path)['model']  # .float().state_dict()
            print(state_dict.keys())
            for key in state_dict.keys():
                if 'block_n' in key:
                    print(state_dict[key].shape)
                    print(state_dict[key])
            state_dict1 = {key[7:]: state_dict[key] for key in state_dict.keys()}  ##去掉DDP的'.module'
            model.load_state_dict(state_dict1, strict=False)
            tflite.update_next_act_scale(model)
            # replace alpha
            # tflite.replace_alpha(model, bit=opt.quantization_bits, check_multiplier=True)
        print('after fusing.................................................')
        print(model)
        # gen int_weight/int_bias
        convert_to_full_int(model)

        ####### 20240129 新增代码 ##############
        for name, layer in model.named_modules():
            if isinstance(layer, (tflite.Conv2d_quantization)):
                layer.inference_type = "all_fp"

        temp_input = torch.zeros([1, 3, 224, 224]).cuda()
        model = model.eval().cuda()
        # # update alpha, act_scale, weight_sacle, M, n
        temp_output = model(temp_input)
        tflite.update_next_act_scale(model)
        del temp_input, temp_output

        for name, layer in model.named_modules():
            if isinstance(layer, (tflite.Conv2d_quantization)):
                layer.inference_type = "full_int"
        ####### 20240129 新增代码 ##############


        # runner.model.module=model

        runner.model = model.cuda()
        # if args.mode == 'evaluate':
        evaluate(runner, loaders)
        # elif args.mode == 'calc_flops':
        flops = tools.get_model_flops(config, runner.get_model())
        print('flops: {}'.format(flops))
        # elif args.mode == 'calc_params':
        params = tools.get_model_parameters(runner.get_model())
        print('params: {}'.format(params))

