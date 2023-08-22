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

# if not (args.compress == 'quant' and not wquant):
#     cache = {}
#     def add_batch(name):
#         def tmp(layer, inp, out):
#             trueobs[name].add_batch(inp[0].data, out.data)
#         return tmp
#     handles = []
#     for name in trueobs:
#         handles.append(layersd[name].register_forward_hook(add_batch(name)))
#     for i in range(args.nrounds):
#         for j, batch in enumerate(dataloader):
#             print(i, j)
#             with torch.no_grad():
#                 run(modeld, batch)
#     for h in handles:
#         h.remove()
#     for name in trueobs:
#         print(name)
#         if args.compress == 'quant':
#             print('Quantizing ...')
#             trueobs[name].quantize()
#         trueobs[name].free()


if aquant:
    print('Quantizing activations ...')
    def init_actquant(name):
        def tmp(layer, inp, out):
            layersp[name].quantizer.find_params(inp[0].data)
        return tmp
    handles = []
    for name in layersd:
        handles.append(layersd[name].register_forward_hook(init_actquant(name)))
    with torch.no_grad():
        run(modeld, next(iter(dataloader)))
    for h in handles:
        h.remove()

if args.save:
    torch.save(modelp.state_dict(), args.save)
    torch.save(modeld.state_dict(), "modeld.pth")

test(modelp, testloader)
