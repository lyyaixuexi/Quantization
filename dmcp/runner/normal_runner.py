# -*- coding:utf-8  -*-

import logging
import os
import sys

import time
import torch
import torch.nn as nn
import utils.distributed as dist
from utils.loss_fn import LabelSmoothCELoss
from utils.meter import AverageMeter, accuracy, \
    calc_adaptive_model_flops, calc_model_parameters


sys.path.append('..')
import tflite_quantization_PACT_weight_and_act as tflite
from overflow_utils import *

def register_hook(model, func):
    # 注册钩子，用于获取每层卷积的输入和输出
    handle_list = []
    for name, layer in model.named_modules():
        if isinstance(layer, tflite.Conv2d_quantization):
            handle = layer.register_forward_hook(func)
            handle_list.append(handle)

    return handle_list
oaq_conv_result = []


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

def hook_conv_results(module, input, output):
    # 获取每层卷积的输出
    oaq_conv_result.append(output.detach().clone())
def overflow_aware(args, batch_size,lr_curr,lr_max,model,batch_idx, oaq_handle_list):
    if (args.OAQ_m!=0) and (batch_idx%args.OAQ_m==0):
                with torch.no_grad():
                    # if dist.get_rank() == 0:
                    #     logging.info("step={}, calculate No and updata alpha!".format(global_step))
                    No = calculate_No(args.gpu, model.module, oaq_conv_result)
                     
                    total_No = tflite.Gather.apply(No)
                    # print('total_No={}'.format(total_No))
                    No = torch.sum(total_No, dim=0)
                    # print('No={}'.format(No))
                    if torch.sum(No).item()!=0 and dist.get_rank()==0:
                        print('overflow still exists')
                    #lr_max = cargs.lr
                    #lr_curr = optimizer.param_groups[0]['lr']
                    update_alpha(model.module, No,  batch_size, lr_max, lr_curr)
                for handle in oaq_handle_list:
                    handle.remove()
                oaq_conv_result.clear()
                
class NormalRunner:
    def __init__(self,args, config, model):
        self.config = config
        self.model = model
        self.logger = logging.getLogger('global_logger')
        self.args=args
        

        self.cur_epoch = 0
        self.cur_step = 0
        self.best_top1 = 0

    def train(self, train_loader, val_loader, optimizer, lr_scheduler, tb_logger):
        print_freq = self.config.logging.print_freq

        flops = calc_adaptive_model_flops(self.model, self.config.dataset.input_size)
        params = calc_model_parameters(self.model)
        self._info('flops: {}, params: {}'.format(flops, params))

        # meters
        batch_time = AverageMeter(print_freq)
        data_time = AverageMeter(print_freq)
        loss_meter = AverageMeter(print_freq)
        top1_meter = AverageMeter(print_freq)
        top5_meter = AverageMeter(print_freq)
        meters = [top1_meter, top5_meter, loss_meter, data_time]
        criterion = self._get_criterion()

        end = time.time()
        for e in range(self.cur_epoch, self.config.training.epoch):
            # train
            if self.config.distributed.enable:
                train_loader.sampler.set_epoch(e)
            for batch_idx, (x, y) in enumerate(train_loader):
                ##################
                if (self.args.OAQ_m!=0) and (batch_idx%self.args.OAQ_m==0):
                    oaq_handle_list = register_hook(self.model.module, hook_conv_results)
                ###################
                self.model.train()
                
                x, y = x.cuda(), y.cuda()
                self._train_one_batch(x, y, optimizer, lr_scheduler, meters, [criterion], end)
                batch_time.update(time.time() - end)
                end = time.time()
                cur_lr = lr_scheduler.get_lr()[0]
                self._logging(tb_logger, e, batch_idx, len(train_loader), meters + [batch_time], cur_lr)
                
                #####################
                if (self.args.OAQ_m!=0) and (batch_idx%self.args.OAQ_m==0):
                    #with torch.no_grad():
                        #overflow_aware(self.args,x.shape[0], cur_lr,self.config.lr_scheduler.base_lr,self.model,batch_idx,oaq_handle_list)
                        batch_size=x.shape[0]
                        if (self.args.OAQ_m!=0) and (batch_idx%self.args.OAQ_m==0):
                            with torch.no_grad():
                                # if dist.get_rank() == 0:
                                #     logging.info("step={}, calculate No and updata alpha!".format(global_step))
                                No = calculate_No(self.args.gpu, self.model.module, oaq_conv_result)

                                total_No = tflite.Gather.apply(No)
                                # print('total_No={}'.format(total_No))
                                No = torch.sum(total_No, dim=0)
                                # print('No={}'.format(No))
                                if torch.sum(No).item()!=0 and dist.get_rank()==0:
                                    print('overflow still exists')
                                lr_max = self.config.lr_scheduler.base_lr
                                #lr_curr = optimizer.param_groups[0]['lr']
                                update_alpha(self.model.module, No,  batch_size, lr_max, cur_lr,self.args.quantization_bits)
                            for handle in oaq_handle_list:
                                handle.remove()
                            oaq_conv_result.clear()
                #####################
                
                # validation
                if  self.cur_step >= self.config.validation.start_val and self.cur_step % self.config.validation.val_freq == 0:
                    print('self.cur_step >= self.config.validation.start_val and self.cur_step % self.config.validation.val_freq == 0')
                    best_top1 = self.best_top1
                    self.validate(val_loader, tb_logger=tb_logger)
                    
                    save_file = self.save(optimizer, e, best_top1=self.best_top1)

                    if self.best_top1 > best_top1:
                        from shutil import copyfile
                        best_file_dir = os.path.join(self.config.save_path, 'best')
                        if not os.path.exists(best_file_dir):
                            os.makedirs(best_file_dir)
                        best_file = os.path.join(best_file_dir, 'best.pth')
                        copyfile(save_file, best_file)
                        print('save best.....................')
                if self.args.Mn_aware:
                    tflite.update_next_act_scale(self.model)
            print('save epoch......')        
            save_file = self.save(optimizer, e, best_top1=self.best_top1)     
                    

    def validate(self, val_loader, tb_logger=None):
        print('normal_runner validate............................')
        batch_time = AverageMeter(0)
        loss_meter = AverageMeter(0)
        top1_meter = AverageMeter(0)
        top5_meter = AverageMeter(0)
        prec1_class_meter = AverageMeter(0)

        def print_bottom_layer_params(module, prefix=""):
            # 如果模块没有子模块，表示为最底层
            if len(list(module.children())) == 0:
                print(f"Bottom Layer: {prefix}")
                # 打印最底层各个参数的数值
                for name, param in module.named_parameters():
                    if 'act_act' in name:
                        print(f"Parameter: {prefix}.{name}, Size: {param.size()}, Values: {param.data}")

            # 递归查找子模块
            for name, child in module.named_children():
                new_prefix = f"{prefix}.{name}" if prefix else name
                print_bottom_layer_params(child, prefix=new_prefix)

        # 假设你已经创建了一个模型实例，例如 model = YourModel()
        # 这里的 YourModel 需要替换为你实际使用的模型

        # 调用递归函数来查找最底层的层并打印出来
        print_bottom_layer_params(self.model)
        self.model.eval()
        self.model.cuda()
        criterion = nn.CrossEntropyLoss()
        end = time.time()
        correct = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.cuda(), y.cuda()
                num = x.size(0)
                
                #oaq_handle_list = register_hook(self.model, hook_conv_results) #这个只能用于训练阶段，不能用于全整型推理阶段
                
                out = self.model(x)
                
                #No = calculate_No(self.args.gpu, self.model, oaq_conv_result)#这个只能用于训练阶段，不能用于全整型推理阶段
                #total_No = tflite.Gather.apply(No)
                #No = torch.sum(total_No, dim=0)
                #print('No={}'.format(No))
                '''
                for handle in oaq_handle_list:
                    handle.remove()
                    oaq_conv_result.clear()
                '''                        
                     
                loss = criterion(out, y.long())
                # top1, top5,prec1_class  = accuracy(out, y, top_k=(1, 3)) ##############################
                pred = out.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(y.data.view_as(pred)).cpu().sum()
                loss_meter.update(loss.item(), num)
                # top1_meter.update(top1.item(), num)
                # top5_meter.update(top5.item(), num)
                # prec1_class_meter.update(prec1_class,num)

                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % self.config.logging.print_freq == 0:
                    self._info('Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'
                               .format(batch_idx, len(val_loader), batch_time=batch_time))

        total_num = torch.tensor([loss_meter.count]).cuda()
        loss_sum = torch.tensor([loss_meter.avg * loss_meter.count]).cuda()
        top1_sum = torch.tensor([top1_meter.avg * top1_meter.count]).cuda()
        top5_sum = torch.tensor([top5_meter.avg * top5_meter.count]).cuda()
        prec1_class_sum = torch.tensor(prec1_class_meter.avg * prec1_class_meter.count).cuda()

        dist.all_reduce(total_num)
        dist.all_reduce(loss_sum)
        dist.all_reduce(top1_sum)
        dist.all_reduce(top5_sum)
        dist.all_reduce(prec1_class_sum)

        val_loss = loss_sum.item() / total_num.item()
        # val_top1 = top1_sum.item() / total_num.item()
        val_top1 = (100. * correct / (len(val_loader.dataset) * 9))
        # val_top5 = top5_sum.item() / total_num.item()
        # val_prec1_class= prec1_class_sum/ total_num.item()

        # self._info('Prec@1 {:.3f}\tPrec@5 {:.3f},Prec1_class {}\tLoss {:.3f}\ttotal_num={}'
        #            .format(val_top1, val_top5, val_prec1_class,val_loss, loss_meter.count))
        # print('Prec@1 {:.3f}\tPrec@5 {:.3f},Prec1_class {}\tLoss {:.3f}\ttotal_num={}'
        #            .format(val_top1, val_top5, val_prec1_class,val_loss, loss_meter.count))

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset) * 9, 100. * correct / (len(val_loader.dataset) * 9)))

        if dist.is_master():
            if val_top1 > self.best_top1:
                self.best_top1 = val_top1
                print('update best_top1:{}'.format(self.best_top1))
                # print('prec1_class:{}'.format(val_prec1_class))


            if tb_logger is not None:
                tb_logger.add_scalar('loss_val', val_loss, self.cur_step)
                tb_logger.add_scalar('acc1_val', val_top1, self.cur_step)
                # tb_logger.add_scalar('acc5_val', val_top5, self.cur_step)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
                val_loss, correct, len(val_loader.dataset) * 9, 100. * correct / (len(val_loader.dataset) * 9)))

    def infer(self, test_loader, train_loader=None):
        self.validate(test_loader)

    @dist.master
    def save(self, optimizer=None, epoch=None, best_top1=None):
        print('normal_runner save......................')
        chk_dir = os.path.join(self.config.save_path, 'checkpoints')
        print(chk_dir)
        if not os.path.exists(chk_dir):
            os.makedirs(chk_dir)
        name = time.strftime('%m%d_%H%M.pth')
        name = os.path.join(chk_dir, name)
        print(name)
        state = {'model': self.model.state_dict()}
        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()
        if epoch is not None:
            state['epoch'] = epoch
            state['cur_step'] = self.cur_step
        if best_top1 is not None:
            state['best_top1'] = best_top1

        torch.save(state, name)
        self._info('model saved at {}'.format(name))
        return name

    def load(self, checkpoint):
        if checkpoint.get('cur_step', None) is not None:
            self.cur_step = checkpoint['cur_step']
        if checkpoint.get('epoch', None) is not None:
            self.cur_epoch = checkpoint['epoch'] + 1

    def get_model(self):
        return self.model

    @dist.master
    def _info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def _get_criterion(self):
        if self.config.training.label_smooth != 'None':
            label_smooth = self.config.training.label_smooth
            criterion = LabelSmoothCELoss(label_smooth, 1000)
            self._info('using label_smooth: {}'.format(label_smooth))
        else:
            criterion = nn.CrossEntropyLoss()

        return criterion

    def _train_one_batch(self, x, y, optimizer, lr_scheduler, meters, criterions, end):
        #print('normal_runner: train_one_batch......')
        top1_meter, top5_meter, loss_meter, data_time = meters
        criterion = criterions[0]
        world_size = dist.get_world_size()

        lr_scheduler.step(self.cur_step)
        self.cur_step += 1
        data_time.update(time.time() - end)

        self.model.zero_grad()
        out = self.model(x)
        loss = criterion(out, y.long())
        loss /= world_size

        top1, top5, prec1_class = accuracy(out, y, top_k=(1, 3))
        
        reduced_loss =dist.all_reduce(loss.clone())
        reduced_top1 =dist.all_reduce(top1.clone(), div=True)
        reduced_top5 =dist.all_reduce(top5.clone(), div=True)
        reduced_top1_class =dist.all_reduce(prec1_class.clone(), div=True)
        #print('prec1_class:{}'.format(prec1_class))

        loss_meter.update(reduced_loss.item())
        top1_meter.update(reduced_top1.item())
        top5_meter.update(reduced_top5.item())

        loss.backward()
        dist.average_gradient(self.model.parameters())
        optimizer.step()

    def _logging(self, tb_logger, epoch_idx, batch_idx, total_batch, meters, cur_lr):
        print_freq = self.config.logging.print_freq
        top1_meter, top5_meter, loss_meter, data_time, batch_time = meters

        if self.cur_step % print_freq == 0 and dist.is_master():
            tb_logger.add_scalar('lr', cur_lr, self.cur_step)
            tb_logger.add_scalar('acc1_train', top1_meter.avg, self.cur_step)
            tb_logger.add_scalar('acc5_train', top5_meter.avg, self.cur_step)
            tb_logger.add_scalar('loss_train', loss_meter.avg, self.cur_step)
            self._info('-' * 80)
            self._info('Epoch: [{0}/{1}]\tIter: [{2}/{3}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                       'LR {lr:.4f}'.format(
                epoch_idx, self.config.training.epoch, batch_idx, total_batch,
                batch_time=batch_time, data_time=data_time, lr=cur_lr))
            self._info('Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                       .format(loss=loss_meter, top1=top1_meter, top5=top5_meter))
