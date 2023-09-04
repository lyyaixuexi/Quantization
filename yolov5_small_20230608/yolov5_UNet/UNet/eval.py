import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from dice_loss import dice_coeff
from unet import UNet
import argparse
from utils.datasets import CityscapesDataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import tflite_quantization_PACT_weight_and_act as tf
import model_transform as mt
import numpy as np
from utils.model_analyse import ModelAnalyse
import os

# hook setting
def register_hook(model, func):
    # 注册钩子，用于获取每层卷积的输入和输出
    handle_list = []
    for name, layer in model.named_modules():
        if isinstance(layer, tf.Conv2d_quantization):
            handle = layer.register_forward_hook(func)
            handle_list.append(handle)

    return handle_list

def hook_conv_results_checkoverflow(module, input, output):
    # conv_accumulator_bits: [min, max], sign=True
    # 检测是否累加器溢出
    bias_bits_global = tf.c_round(module.bias_bits)
    min = - 2 ** (bias_bits_global - 1) + 1
    max = 2 ** (bias_bits_global - 1) - 1
    if isinstance(module, tf.Conv2d_quantization):
        scale = (module.act_scale * module.weight_scale).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(
            output)
    output = tf.c_round(output.detach() / scale)
    down_overflow_index = output < min
    up_overflow_index = output > max
    No = (torch.sum(down_overflow_index) + torch.sum(up_overflow_index))

    if No>0:
        print("############################ overflow happen ###################")
        print("overflow No: {}".format(No))
        print("overflow module: {}".format(module))
        print("module.alpha: {}".format(module.alpha))
        print("module.act_scale: {}".format(module.act_scale))
        print("output.min: {}".format(output.min()))
        print("output.max: {}".format(output.max()))
        print("max: {}".format(max))

def eval_net(net, loader, device, input_scale):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    pix_tot=0
    I = [0] * net.n_classes
    U = [0] * net.n_classes
    dice_I = [0] * net.n_classes
    dice_U = [0] * net.n_classes

    if args.check_overflow:
        # Set hook to store Conv results for No
        print("Set hook to store Conv results for No")
        handle_list = register_hook(net, hook_conv_results_checkoverflow)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for image, targetmask in loader:
            imgs, true_masks = image, targetmask
            imgs = imgs.to(device=device, dtype=torch.float32)
            if args.inference_type=='full_int':
                imgs=imgs/input_scale
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                # tot += F.cross_entropy(mask_pred, true_masks).item()

                # class_mask=mask_pred.argmax(dim=1)
                # class_mask = class_mask.unsqueeze(0).float()
                # class_mask = F.interpolate(class_mask, size=[1024, 2048], mode='nearest')
                # class_mask = class_mask.squeeze(0)

                class_mask = mask_pred.argmax(dim=1)
                class_mask = TF.resize(class_mask, size=(1024, 2048), interpolation=TF.InterpolationMode.NEAREST)

            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()

            for i in range(net.n_classes):
                I[i]+=torch.sum(torch.where((class_mask == i) * (true_masks == i),torch.ones_like(true_masks),torch.zeros_like(true_masks))).item()
                dice_I[i]+=2*torch.sum(torch.where((class_mask == i) * (true_masks == i),torch.ones_like(true_masks),torch.zeros_like(true_masks))).item()

                U[i]+=torch.sum(torch.where((class_mask == i) + (true_masks == i),torch.ones_like(true_masks),torch.zeros_like(true_masks))).item()
                dice_U[i]+=torch.sum(torch.where(class_mask == i,torch.ones_like(true_masks),torch.zeros_like(true_masks))).item()+torch.sum(torch.where(true_masks == i,torch.ones_like(true_masks),torch.zeros_like(true_masks))).item()

            pix_batch=torch.sum(torch.where(class_mask==true_masks,torch.ones_like(true_masks),torch.zeros_like(true_masks))).item()
            pix_tot+=pix_batch
            pbar.update()

    if args.check_overflow:
        for handle in handle_list:
            handle.remove()

    mpa = pix_tot / n_val / true_masks.size()[0] / true_masks.size()[1] / true_masks.size()[2]
    logging.info('mpa: {}%'.format(mpa*100))
    logging.info('print IoU:')
    K=0
    IoU_tot=0
    for i in range(net.n_classes):
        if U[i]!=0:
            K+=1
            IoU_tot += I[i]/U[i]
            print('{}:{}'.format(i,I[i]/U[i]))
        else:
            print('class {} not exist'.format(i))
    print('mIoU: {}\n'.format(IoU_tot/K))
    # logging.info('print dice:\n')
    # for i in range(net.n_classes):
    #     print('{}:{}'.format(i,dice_I[i]/dice_U[i] if dice_U[i]!=0 else 0))

    net.train()

    return IoU_tot/K


def convert_to_full_int(net, args, device):
    # inference one time to update M and n
    net.eval()
    temp_input = torch.zeros([1, 3, 256, 512], device=device)
    # update alpha, act_scale, weight_sacle for 魔改
    temp_output = net(temp_input)
    #　update next_act_scale for 魔改
    tf.replace_next_act_scale(net)
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
        if isinstance(layer, (tf.Conv2d_quantization)):
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

    next_act_list = tf.layer_transform(args, net)

    return act_scale_list, next_act_list


def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-m', '--model', dest='model', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-t', '--type', type=str, default='',
                        help='type of model')

    parser.add_argument('-x', '--quantization_bits', metavar='QB', type=int, nargs='?', default=6,
                        help='quantization_bits', dest='quantization_bits')
    parser.add_argument('-y', '--m_bits', metavar='MB', type=int, nargs='?', default=12,
                        help='m_bits', dest='m_bits')
    parser.add_argument('-z', '--bias_bits', metavar='BB', type=int, nargs='?', default=16,
                        help='bias_bits', dest='bias_bits')

    parser.add_argument('--inference_type', type=str, default='', help='full_int, all_fp')
    parser.add_argument('--check_overflow', action='store_true',
                        help='if true, check the overflow of convolution before training')

    parser.add_argument('--save_model', action='store_true',
                        help='if true, save the model ')
    parser.add_argument("--result_name", type=str, default='data/output_int8', help='results path of transformed_model')

    parser.add_argument('--half_channel', action='store_true',
                        help='if true, use half_channel')

    parser.add_argument('--quarter_channel', action='store_true',
                        help='if true, use quarter_channel')

    parser.add_argument('--strict_cin_number', action='store_true',
                        help='if true, use half_channel')

    parser.add_argument('--mixture_16bit_8bit', default="mixture_quant_config.txt", type=str,
                        help="whether to use mixture quantization of 16bit and 8bit")

    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args=get_args()

    net = UNet(n_channels=3, n_classes=20, nearest=True, half_channel=args.half_channel, quarter_channel=args.quarter_channel, strict_cin_number=args.strict_cin_number)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model:
        if args.type == '':
            pass
        if 'q' in args.type:
            # code for mixture bit quantization
            if args.mixture_16bit_8bit is not None:
                bits_setting = [16, 8]
                print("use mixture quantization of 16bit and 8bit")
                if os.path.isfile(args.mixture_16bit_8bit):
                    # load the config from a file
                    print("load quant config from: {}".format(args.mixture_16bit_8bit))
                    with open(args.mixture_16bit_8bit, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f.readlines()]
                        mixture_quant_config = {}
                        for line in lines:
                            mixture_quant_config[line.split(" ")[0]] = line.split(" ")[1]

                    layer_bit_dict = {}
                    for param_name in mixture_quant_config.keys():
                        layer_bit_dict[param_name[:-7]] = int(mixture_quant_config[param_name][:-3])

                else:
                    print("args.mixture_16bit_8bit is not a file!")
                    assert False

                net = tf.replace(net, args.quantization_bits, args.m_bits, args.bias_bits, args.inference_type, False, False,
                                 layer_bit_dict=layer_bit_dict)

            else:
                net = tf.replace(net, args.quantization_bits, args.m_bits, args.bias_bits, args.inference_type, False,
                                 False)
        if 'f' in args.type:
            net = tf.fuse_doubleconv(net)
        if 'm' in args.type:
            net = tf.open_Mn(net, args.m_bits)

    net.load_state_dict(torch.load(args.model))
    logging.info(f'Model loaded from {args.model}')

    print(net)

    compressed_model_size = os.path.getsize(args.model) / 1024 / 1024  # convert number of Byte to MB
    print("the size of: {} is:{:.2f} MB".format(args.model, compressed_model_size))
    
    if args.inference_type != 'full_int':
        model_analyse = ModelAnalyse(net.to(device))
        model_analyse.params_count()
        model_analyse.flops_compute(torch.randn(1, 3, 256, 512).to(device))

    input_scale = None
    if args.inference_type == 'full_int':
        net, input_scale=tf.layer_transform(args, net)
        input_scale = input_scale.to(device)

    if args.save_model:
        quantized_param_file = args.result_name + '_param'
        scale_table_file = args.result_name + '.scale'
        convert_to_full_int(net, args, device=torch.device('cpu'))
        with torch.no_grad():
            mt.model_transform(net, quantized_param_file, scale_table_file)

        print("finish save integer model!!!")
        print("save param in:{}".format(quantized_param_file))
        print("save scale in:{}".format(scale_table_file))
        assert False

    logging.info(f'Using device {device}')
    net = net.to(device=device)

    train = CityscapesDataset('/gdata/Cityscapes', split='train', mode='fine', augment=False)
    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True,drop_last=True)

    # test=CityscapesDataset('~/Pytorch-UNet/data',split='test',mode='fine',augment=False)
    # test_loader = DataLoader(test, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True,drop_last=True)

    val = CityscapesDataset('/gdata/Cityscapes', split='val', mode='fine', augment=False)
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True,drop_last=False)
    print('input_scale={}'.format(input_scale))
    # eval_net(
    #     net=net,
    #     loader=train_loader,
    #     device=device,
    #     input_scale=input_scale
    # )
    eval_net(
        net=net,
        loader=val_loader,
        device=device,
        input_scale=input_scale
    )
    # eval_net(
    #     net=net,
    #     loader=test_loader,
    #     device=device
    # )









