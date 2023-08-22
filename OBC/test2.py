import argparse
import copy
import os

import torch
import torch.nn as nn

from datautils import *
from modelutils import *
from quant import *
from trueobs import *


parser = argparse.ArgumentParser()

parser.add_argument('model', type=str)
parser.add_argument('dataset', type=str)
parser.add_argument(
    'compress', type=str, choices=['quant', 'nmprune', 'unstr', 'struct', 'blocked']
)
parser.add_argument('--load', type=str, default='')
parser.add_argument('--datapath', type=str, default='')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save', type=str, default='')

parser.add_argument('--nsamples', type=int, default=1024)
parser.add_argument('--batchsize', type=int, default=-1)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--nrounds', type=int, default=-1)
parser.add_argument('--noaug', action='store_true')

parser.add_argument('--wbits', type=int, default=32)
parser.add_argument('--abits', type=int, default=32)
parser.add_argument('--wperweight', action='store_true')
parser.add_argument('--wasym', action='store_true')
parser.add_argument('--wminmax', action='store_true')
parser.add_argument('--asym', action='store_true')
parser.add_argument('--aminmax', action='store_true')
parser.add_argument('--rel-damp', type=float, default=0)

parser.add_argument('--prunen', type=int, default=2)
parser.add_argument('--prunem', type=int, default=4)
parser.add_argument('--blocked_size', type=int, default=4)
parser.add_argument('--min-sparsity', type=float, default=0)
parser.add_argument('--max-sparsity', type=float, default=0)
parser.add_argument('--delta-sparse', type=float, default=0)
parser.add_argument('--sparse-dir', type=str, default='')

args = parser.parse_args()

dataloader, testloader = get_loaders(
    args.dataset, path=args.datapath,
    batchsize=args.batchsize, workers=args.workers,
    nsamples=args.nsamples, seed=args.seed,
    noaug=args.noaug
)
if args.nrounds == -1:
    args.nrounds = 1 if 'yolo' in args.model or 'bert' in args.model else 10
    if args.noaug:
        args.nrounds = 1
get_model, test, run = get_functions(args.model)

aquant = args.compress == 'quant' and args.abits < 32
wquant = args.compress == 'quant' and args.wbits < 32

modelp = get_model()
if args.compress == 'quant' and args.load:
    modelp.load_state_dict(torch.load(args.load))
if aquant:
    add_actquant(modelp)
modeld = get_model()
layersp = find_layers(modelp)
layersd = find_layers(modeld)

trueobs = {}
for name in layersp:
    layer = layersp[name]
    if isinstance(layer, ActQuantWrapper):
        layer = layer.module
    trueobs[name] = TrueOBS(layer, rel_damp=args.rel_damp)
    if aquant:
        layersp[name].quantizer.configure(
            args.abits, sym=args.asym, mse=not args.aminmax
        )
    if wquant:
        trueobs[name].quantizer = Quantizer()
        trueobs[name].quantizer.configure(
            args.wbits, perchannel=not args.wperweight, sym=not args.wasym, mse=not args.wminmax
        )


modelp.load_state_dict(torch.load('/mnt/cephfs/home/lyy/Quantization/OBC/rn18_4w4a.pth', map_location='cpu'))

for name, param in modelp.named_parameters():
    print(f"{name}: {param}")
