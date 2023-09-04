import argparse
import logging
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from unet import UNet
import tflite_quantization_PACT_weight_and_act as tf

mapping={
            0: 0,  # unlabeled
            1: 0,  # ego vehicle
            2: 0,  # rect border
            3: 0,  # out of roi
            4: 0,  # static
            5: 0,  # dynamic
            6: 0,  # ground
            7: 1,  # road
            8: 2,  # sidewalk
            9: 0,  # parking
            10: 0,  # rail track
            11: 3,  # building
            12: 4,  # wall
            13: 5,  # fence
            14: 0,  # guard rail
            15: 0,  # bridge
            16: 0,  # tunnel
            17: 6,  # pole
            18: 0,  # polegroup
            19: 7,  # traffic light
            20: 8,  # traffic sign
            21: 9,  # vegetation
            22: 10,  # terrain
            23: 11,  # sky
            24: 12,  # person
            25: 13,  # rider
            26: 14,  # car
            27: 15,  # truck
            28: 16,  # bus
            29: 0,  # caravan
            30: 0,  # trailer
            31: 17,  # train
            32: 18,  # motorcycle
            33: 19,  # bicycle
            -1: 0  # licenseplate
        }
mappingrgb={
            0: (0, 0, 0),  # unlabeled
            1: (0, 0, 0),  # ego vehicle
            2: (0, 0, 0),  # rect border
            3: (0, 0, 0),  # out of roi
            4: (0, 0, 0),  # static
            5: (111, 74,  0),  # dynamic
            6: (81,  0, 81),  # ground
            7: (128, 64,128),  # road
            8: (244, 35,232),  # sidewalk
            9: (250,170,160),  # parking
            10: (230,150,140),  # rail track
            11: (70, 70, 70),  # building
            12: (102,102,156),  # wall
            13: (190,153,153),  # fence
            14: (180,165,180),  # guard rail
            15: (150,100,100),  # bridge
            16: (150,120, 90),  # tunnel
            17: (153,153,153),  # pole
            18: (153,153,153),  # polegroup
            19: (250, 170, 30),  # traffic light
            20: (220, 220,  0),  # traffic sign
            21: (107, 142, 35),  # vegetation
            22: (152, 251,152),  # terrain
            23: (70,130,180),  # sky
            24: (220, 20, 60),  # person
            25: (255, 0,  0),  # rider
            26: (0, 0,142),  # car
            27: (0, 0, 70),  # truck
            28: (0, 60, 100),  # bus
            29: (0, 0, 90),  # caravan
            30: (0, 0, 110),  # trailer
            31: (0, 80, 100),  # train
            32: (0, 0, 230),  # motorcycle
            33: (119, 11, 32),  # bicycle
            -1: (0,  0, 142)  # licenseplate
}

def predict_img(net,
                img,
                device):
    net.eval()

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        pred_mask = probs.argmax(dim=1)

        pred_mask = TF.resize(pred_mask, size=(1024,2048), interpolation=TF.InterpolationMode.NEAREST)

    return pred_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('-t', '--type', type=str, default='',
                        help='type of model')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)

    parser.add_argument('-x', '--quantization_bits', metavar='QB', type=int, nargs='?', default=6,
                        help='quantization_bits', dest='quantization_bits')
    parser.add_argument('-y', '--m_bits', metavar='MB', type=int, nargs='?', default=12,
                        help='m_bits', dest='m_bits')
    parser.add_argument('-z', '--bias_bits', metavar='BB', type=int, nargs='?', default=16,
                        help='bias_bits', dest='bias_bits')

    parser.add_argument('--inference_type', type=str, default='', help='full_int, all_fp')

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray(mask.astype(np.uint8))

def class_to_rgb(mask):
    assert mask.dim() == 2
    mask2class={}
    for v,k in mapping.items():
        if k!=0:
            mask2class[k]=v
        mask2class[0]=0
    rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
    for k in mask2class:
        for i in range(3):
            rgbimg[i][mask == k] = mappingrgb[mask2class[k]][i]
    return rgbimg

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=20, nearest=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model:
        if args.type=='':
            pass
        if 'q' in args.type:
            net = tf.replace(net, args.quantization_bits, args.m_bits, args.bias_bits, args.inference_type, False, False)
        if 'f' in args.type:
            net = tf.fuse_doubleconv(net)
        if 'm' in args.type:
            net=tf.open_Mn(net)

    net.load_state_dict(torch.load(args.model))
    logging.info(f'Model loaded from {args.model}')

    logging.info(f'Using device {device}')
    net.to(device=device)
    print(net)

    if 'm' in args.type:
        tf.replace_next_act_scale(net)

    input_scale=None
    if args.inference_type == 'full_int':
        with torch.no_grad():
            net, input_scale=tf.layer_transform(args, net)
            print('input_scale={}'.format(input_scale))

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        img = img.convert('RGB')
        img = TF.resize(img, size=(256, 512), interpolation=TF.InterpolationMode.NEAREST)
        img = TF.to_tensor(img)
        if args.inference_type=='full_int':
            img=img.to(device)/input_scale

        class_mask = predict_img(net=net, img=img, device=device)
        class_mask=class_mask.squeeze()
        rgbimg=class_to_rgb(class_mask).numpy().transpose((1,2,0))

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(rgbimg)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))
