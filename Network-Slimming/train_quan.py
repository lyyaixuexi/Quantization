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
                                                            inference_type="all_fp",
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
best_prec1 = 0.
def save_checkpoint(state, is_best,filename, model_best):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_best)

for epoch in range(start_epoch, args.epochs):  # epoch ------------------------------------------------------------------
    print("epoch: ", epoch)
    model.train()
    epoch_loss = 0
    adjust_learning_rate(optimizer, args.lr, epoch)


    pbar = enumerate(train_loader)

    pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    OAQ_m = 10
    optimizer.zero_grad()
    for i, (data, targets) in pbar:  # batch -------------------------------------------------------------
        if (OAQ_m != 0) and (global_step % OAQ_m == 0):
            oaq_handle_list = register_hook(model, hook_conv_results)


        # ni = i + nb * epoch  # number integrated batches (since train start)
        data = data.to(device, non_blocking=True)
        targets = targets.to(device)
        # Forward
        # with amp.autocast(enabled=cuda):
        output = model(data)  # forward
        # loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
        loss = cls_multi_patch_loss(output, targets).cuda()

        epoch_loss += loss.item()

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        if (OAQ_m != 0) and (global_step % OAQ_m == 0):
            with torch.no_grad():
                # if dist.get_rank() == 0:
                #     logging.info("step={}, calculate No and updata alpha!".format(global_step))
                No = calculate_No(device, model, oaq_conv_result, logging)
                # total_No = tflite_quantization_PACT_weight_and_act.Gather.apply(No)
                # print('total_No={}'.format(total_No))
                # No = torch.sum(total_No, dim=0)
                print('\nNo={}'.format(No))
                # if torch.sum(No).item()!=0 and dist.get_rank()==0:
                if torch.sum(No).item() != 0:
                    print('overflow still exists')
                lr_max = 0.01
                lr_curr = optimizer.param_groups[0]['lr']
                update_alpha(device, model, No, args.batch_size, lr_max, lr_curr, logging)

            for handle in oaq_handle_list:
                handle.remove()

            oaq_conv_result.clear()

        global_step += 1


    prec1 = test(model)

    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, os.path.join(args.save, args.filename + '.pth'),
        os.path.join(args.save, args.filename + '_best.pth'))

    save_checkpoint(model, is_best, os.path.join(args.save, args.filename + '_total_model.pth'),
        os.path.join(args.save, args.filename + '_best_total_model.pth'))

# model = tflite_quantization_PACT_weight_and_act.replace(model=model,
#                                                         quantization_bits=8,
#                                                         m_bits=12,
#                                                         bias_bits=16,
#                                                         inference_type="full_int",
#                                                         Mn_aware=True)
# model.cuda()
# pre1 = test(model)
# torch.save(model, os.path.join(args.save, args.filename+'total_model.pth'))
# torch.save(model.state_dict(), os.path.join(args.save, args.filename+'state_dict_model.pth'))
        # end batch ------------------------------------------------------------------------------------------------





#     print("lr_max: ", lr_max)
#     print("lr_curr: ", lr_curr)
#     for name, layer in model.named_modules():
#         if (name == 'model.24.m.2'):
#             print('layer.alpha: ', layer.alpha)
#
#     for handle in handle_list:
#         handle.remove()
#
#     # Scheduler
#     lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
#     scheduler.step(epoch_loss)
#
#     # =============== show bn weights ===================== #
#     module_list = []
#     # module_bias_list = []
#     for i, layer in model.named_modules():
#         if isinstance(layer, nn.BatchNorm2d) and i not in ignore_bn_list:
#             bnw = layer.state_dict()['weight']
#             bnb = layer.state_dict()['bias']
#             module_list.append(bnw)
#             # module_bias_list.append(bnb)
#             # bnw = bnw.sort()
#             # print(f"{i} : {bnw} : ")
#     size_list = [idx.data.shape[0] for idx in module_list]
#
#     bn_weights = torch.zeros(sum(size_list))
#     bnb_weights = torch.zeros(sum(size_list))
#     index = 0
#     for idx, size in enumerate(size_list):
#         bn_weights[index:(index + size)] = module_list[idx].data.abs().clone()
#         # bnb_weights[index:(index + size)] = module_bias_list[idx].data.abs().clone()
#         index += size
#
#     if RANK in [-1, 0]:
#         # mAP
#         callbacks.run('on_train_epoch_end', epoch=epoch)
#         ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
#         final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
#         if not noval or final_epoch:  # Calculate mAP
#             results, maps, _ = val.run(data_dict,
#                                        batch_size=batch_size // WORLD_SIZE * 2,
#                                        imgsz=imgsz,
#                                        model=ema.ema,
#                                        single_cls=single_cls,
#                                        dataloader=val_loader,
#                                        save_dir=save_dir,
#                                        plots=False,
#                                        callbacks=callbacks,
#                                        compute_loss=compute_loss)
#
#         # Update best mAP
#         fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
#         if fi > best_fitness:
#             best_fitness = fi
#         log_vals = list(mloss) + list(results) + lr + [0]
#         callbacks.run('on_fit_epoch_end', log_vals, bn_weights.numpy(), epoch, best_fitness, fi)
#
#         # Save model
#         if (not nosave) or (final_epoch and not evolve):  # if save
#             ckpt = {'epoch': epoch,
#                     'best_fitness': best_fitness,
#                     # 'model': deepcopy((model)).half(),
#                     'ema': deepcopy(ema.ema),
#                     'model': model,
#                     # 'ema': ema,
#                     'updates': ema.updates,
#                     'optimizer': optimizer.state_dict(),
#                     'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
#                     'date': datetime.now().isoformat()}
#
#             # Save last, best and delete
#             torch.save(ckpt, last)
#             if best_fitness == fi:
#                 torch.save(ckpt, best)
#             if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
#                 torch.save(ckpt, w / f'epoch{epoch}.pt')
#             del ckpt
#             callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)
#
#         # Stop Single-GPU
#         if RANK == -1 and stopper(epoch=epoch, fitness=fi):
#             break
#
#         # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
#         # stop = stopper(epoch=epoch, fitness=fi)
#         # if RANK == 0:
#         #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks
#
#     # Stop DPP
#     # with torch_distributed_zero_first(RANK):
#     # if stop:
#     #    break  # must break all DDP ranks
#
#     # end epoch ----------------------------------------------------------------------------------------------------
# # end training -----------------------------------------------------------------------------------------------------



