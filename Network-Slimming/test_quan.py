from __future__ import print_function
import tflite_quantization_PACT_weight_and_act
import os
import json
import argparse
import shutil
import torch
import torch.nn as nn
from torch.utils import data
import models
import math
from apex import amp
from models.RanGer import Ranger
from models.Label_SmoothCELoss import LabelSmoothCELoss, LabelSmoothCELoss_modify
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from dataloader_mulit_patch import img_cls_by_dir_loader as data_loader
from lr_scheduling import *
from tqdm import tqdm
import logging
import model_transform as mt


seed = 42
torch.manual_seed(seed)

oaq_conv_result = []
def hook_conv_results(module, input, output):
    # 获取每层卷积的输出
    oaq_conv_result.append(output.detach().clone())


def register_hook(model, func):
    # 注册钩子，用于获取每层卷积的输入和输出
    handle_list = []
    for name, layer in model.named_modules():
        # if isinstance(layer, tf.Conv2d_quantization):
        if(name=='conv9'):
            handle = layer.register_forward_hook(func)
            handle_list.append(handle)

    return handle_list

def adjust_learning_rate(optimizer, init_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr



def update_alpha(device, model, No, iteration_batch_size, lr_max, lr_curr, logger):
    # logger.info("update alpha: start!")

    # merge No from every GPU
    # logger.info('before merge, No={}'.format(No))
    # No = AllReduce_overflow.apply(No)
    # logger.info('After merge, No={}'.format(No))

    index = 0
    for name, layer in model.named_modules():
        # if isinstance(layer, tflite.Conv2d_quantization):
        if(name=='conv9'):
            # logger.info('No[{}]={}, iteration_batch_size={}, lr_max={}, lr_curr={}'.format(index, No[index], iteration_batch_size, lr_max, lr_curr))
            if No[index] > 0:
                # v1: better
                update_value = torch.min((lr_curr * torch.log( (No[index] / iteration_batch_size) + 1 )), torch.Tensor([lr_max])[0].to(device))
                # v2
                # update_value = torch.min((lr_curr * torch.log(No[index])), torch.Tensor([lr_max])[0].to(device))
                # layer.alpha += update_value

                # if index==0:
                #     layer.alpha += update_value * 5000
                # else:
                #     layer.alpha += update_value * 100

                layer.alpha += update_value * 100
                # layer.alpha += update_value

            elif No[index] == 0:
                pass
            #     lr_curr_gpu = torch.Tensor([lr_curr])[0].to(device)
            #     layer.alpha -= lr_curr_gpu

            else:
                assert False, logger.info('No[{}] ={} impossible !!!'.format(index, No[index]))
            index += 1
            logger.info('index = {}  After update, alpha={}'.format(index, layer.alpha))
            print(('index = {}  After update, alpha={}'.format(index, layer.alpha)))
            # print("weight", layer.weight_scale)
            # print("act", layer.act_scale)


def calculate_No(device, model, oaq_conv_result, logger):
    # logger.info("calculate No: start!")
    # logger.info('len(oaq_conv_result)={}'.format(len(oaq_conv_result)))

    index = 0
    No = torch.zeros(len(oaq_conv_result), device=device)  # nx1, n: the number of conv layer
    for name, layer in model.named_modules():
        if(name=='conv9'):
        # if isinstance(layer, tflite.Conv2d_quantization):
            # oaq_conv_result[index]: batch*C_out*h*w
            # layer.scale_int_weight: [C_out]
            bias_bits_global = tflite_quantization_PACT_weight_and_act.c_round(layer.bias_bits)
            min = - 2 ** (bias_bits_global - 1) + 1
            max = 2 ** (bias_bits_global - 1) - 1
            if (layer.Mn_aware == False):
                scale = (layer.act_scale * layer.weight_scale).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(oaq_conv_result[index])
                oaq_conv_result[index] = tflite_quantization_PACT_weight_and_act.c_round(oaq_conv_result[index] / scale)
            else:
                oaq_conv_result[index] = layer.tmp_int_output2
            down_overflow_index = oaq_conv_result[index] < min
            up_overflow_index = oaq_conv_result[index] > max
            No[index] = (torch.sum(down_overflow_index) + torch.sum(up_overflow_index)).to(device)
            index += 1

    if index != len(oaq_conv_result):
        assert False, logger.info('Conv2d_quantization number != len(oaq_conv_result)')
    # print("No for each layer:{}".format(No))
    return No

dict={}
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
#parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])

parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--using-amp_loss', '-amp_loss', dest='amp_loss', action='store_true',
                    help='using amp_loss mix f16 and f32')
parser.add_argument('--s', type=float, default=1e-5,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 0.010)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--quant', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be quant')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--arch', default='resnet', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=20, type=int,
                    help='depth of the neural network')
parser.add_argument('--warm_up_epochs', default=5, type=int,
                    help='warm_up_epochs of the neural network')
parser.add_argument('--save', default='./logs/',
                    help='checkpoint__save')
parser.add_argument('--filename', default='',
                    help='filename__save')
parser.add_argument('--quantization_bits', default=8, type=int,
                    help='quan_bits')
parser.add_argument('--m_bits', default=12, type=int,
                    help='m_bits')
parser.add_argument('--bias_bits', default=16, type=int,
                    help='acc_bits')
parser.add_argument('-M', '--Mn_aware', action='store_true',
                        help="Mn_aware", default=False)
parser.add_argument('--save_quantized_layer', action='store_true', help='if true, save the quantized layer')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

with open('./train_list_test.txt', 'r') as f:
    lines = f.readlines()
    lines = [i.strip('\n') for i in lines]

root_dir = lines

args.multi_patch = True
args.img_size = 288
args.color_mode = 'YUV_bt601V'

train_dataset = data_loader(root_dir, split="train", is_transform=True, img_size=args.img_size, color_mode=args.color_mode, multi_patch = args.multi_patch)
train_loader = data.DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True
)

val_dataset = data_loader(root_dir, split="val", is_transform=True, img_size=args.img_size, color_mode=args.color_mode, multi_patch = args.multi_patch)
test_loader = data.DataLoader(
    val_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True
)

if args.Mn_aware == False:
    print("loading.....")
    args.quant = args.save + args.quant + '.pth'
    checkpoint = torch.load(args.quant)
    args.refine = args.save + args.refine + '.pth'
    checkpoint1 = torch.load(args.refine)
    from models.traffic_sign_cls_1_modify import traffic_sign_cls_modify as ClassifyNet
    model = ClassifyNet(cfg=checkpoint1['cfg'])
    # model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
    model = tflite_quantization_PACT_weight_and_act.replace(model=model,
                                                            quantization_bits=args.quantization_bits,
                                                            m_bits=args.m_bits,
                                                            bias_bits=args.bias_bits,
                                                            inference_type="all_fp",
                                                            Mn_aware=args.Mn_aware)

else:
    # model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    from models.traffic_sign_cls_1_modify import traffic_sign_cls_modify as ClassifyNet
    args.quant = args.save + args.quant + '.pth'
    checkpoint = torch.load(args.quant)
    args.refine = args.save + args.refine + '.pth'
    checkpoint1 = torch.load(args.refine)
    from models.traffic_sign_cls_1_modify import traffic_sign_cls_modify as ClassifyNet

    model = ClassifyNet(cfg=checkpoint1['cfg'])
    model = tflite_quantization_PACT_weight_and_act.replace(model=model,
                                                            quantization_bits=args.quantization_bits,
                                                            m_bits=args.m_bits,
                                                            bias_bits=args.bias_bits,
                                                            inference_type="full_int",
                                                            Mn_aware=args.Mn_aware)
    print(checkpoint.keys())
    if 'best_prec1' in checkpoint.keys():
        print(checkpoint['best_prec1'])
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

if args.cuda:
    model.cuda()

model, _ = tflite_quantization_PACT_weight_and_act.layer_transform(model)



freeze_list = ['conv9.weight', 'conv9.bias', 'conv9.act_max']

model.cuda()

for k, v in model.named_parameters():
    v.requires_grad = False
    if any(x in k for x in freeze_list):
        print(f'only train {k}')
        v.requires_grad = True

# print(model)

g0, g1, g2 = [], [], []  # optimizer parameter groups

for v in model.modules():
    if isinstance(v, tflite_quantization_PACT_weight_and_act.Conv2d_quantization):
        g2.append(v.bias)
        g1.append(v.weight)
print(len(g1))
print(len(g2))

# optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=1e-8, momentum=0.9)
#
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-8)
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
    weight_decay=5e-4,
    momentum=0.9,
)

start_epoch, best_fitness = 0, 0.0

epochs = 25

nb = len(train_loader)  # number of batches

global_step = 0

device = "cuda:0"

def cls_multi_patch_loss(pred, target):
    '''
    This loss is for the mutli patch concat input
    '''
    batchsize, c_num, _, _ = pred.shape
    pred = pred.permute(0, 2, 3, 1)
    pred = pred.reshape(-1, c_num)
    # pred = pred.permute(0, 2, 1)
    # pred = pred.reshape(-1)
    target = target.reshape(-1)
    loss_cls = torch.nn.functional.cross_entropy(pred, target.long())
    return loss_cls

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = data, target
            output = model(data)

            # batchsize, c_num, _, _ = output.shape
            # output = output.permute(0, 2, 3, 1)
            # output = output.reshape(-1, c_num)
            # target = target.reshape(-1)

            # criterion = LabelSmoothCELoss_modify().cuda()
            # test_loss += criterion(output, target).item() # sum up batch loss
            test_loss += cls_multi_patch_loss(output, target).cuda().item()

            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset)*9,
        100. * correct / (len(test_loader.dataset)*9)))
    return correct / float((len(test_loader.dataset)*9))


pre1 = test(model)


def convert_to_full_int(net, device):
    # inference one time to update M and n
    net.eval()

    # temp_input = torch.zeros([1, 3, 256, 512], device=device)
    temp_input = torch.zeros([1, 3, 416, 416], device=device)

    # update alpha, act_scale, weight_sacle for 魔改
    # temp_output = net(temp_input)
    # 　update next_act_scale for 魔改
    # tf.replace_next_act_scale(net)
    # update M n for 魔改
    temp_output = net(temp_input)
    del temp_input, temp_output

    # 　replace float param to interger param
    index = 0
    act_scale_list = []
    weight_scale_list = []
    m_bits_list = []
    layer_list = []
    for name, layer in net.named_modules():
        if isinstance(layer, (tflite_quantization_PACT_weight_and_act.Conv2d_quantization)):
            act_scale_list.append(layer.act_scale)
            weight_scale_list.append(layer.weight_scale)
            m_bits_list.append(layer.m_bits)
            layer_list.append(layer)
            layer.inference_type = "full_int"
            layer.index = index
            index += 1

            # print some information
            print("act_bits:{}  weight_bits:{} bias_bits:{}".format(layer.act_bits, layer.weight_bits,
                                                                    layer.bias_bits))

    next_act_list = tflite_quantization_PACT_weight_and_act.layer_transform(net)

    return act_scale_list, next_act_list



if args.save_quantized_layer:
    model.to(device='cpu')
    # result_name="output_full_int/output_w4a4b16"
    quantized_param_file = args.filename + '_param'
    scale_table_file = args.filename + '.scale'
    convert_to_full_int(model, device=torch.device('cpu'))
    with torch.no_grad():
        mt.model_transform(model, quantized_param_file, scale_table_file)

    print("finish save integer model!!!")
    print("save param in:{}".format(quantized_param_file))
    print("save scale in:{}".format(scale_table_file))











