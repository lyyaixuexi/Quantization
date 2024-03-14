# -*- coding:utf-8  -*-

import numpy as np
import torch
import torch.nn as nn


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length

        self.history = None
        self.count, self.sum = None, None
        self.val, self.avg = None, None

        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def accuracy(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        #print(top_k)
        max_k = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        
        #print(target.shape)
        #print('pred')
        #print(pred.shape)
        
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        #print('correct')
        #print(correct.shape)

        res = []
        '''
        
        print(type(correct))
        print(correct.device)
        print(correct)
        '''
        precision=[1e-5,1e-5,1e-5]#,1e-5,1e-5,1e-5,1e-5]
        correct_1 = correct[:1]#.contiguous().view(-1).float()
        for i in range(max_k):
            pos=torch.nonzero(target==i)
            nlabel=len(pos)
            b=(target==i) 
            prod=correct_1*b #得到所有为第i类且正确的0/1向量
            if nlabel>0:
                prec=prod.sum()/nlabel*100.0
                precision[i]=prec
              
        for k in top_k:
            #print(k)
            #print(correct[:k])
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        if len(top_k)>1:
            return res[0],res[1],torch.tensor(precision,dtype=torch.float)
        else:
            return res[0], torch.tensor(precision,dtype=torch.float)
    


def calc_model_flops(model, input_size, mul_add=False):
    #print('meter:calc_model_flops')
    hook_list = []
    module_flops = []

    def conv_hook(self, input, output):
        output_channels, output_height, output_width = output[0].size()
        bias_ops = 1 if self.bias is not None else 0
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.cur_in_ch / self.cur_group)
        flops = (kernel_ops * (2 if mul_add else 1) + bias_ops) * output_channels * output_height * output_width
        module_flops.append(flops)

    def linear_hook(self, input, output):
        weight_ops = self.weight.nelement() * (2 if mul_add else 1)
        bias_ops = self.bias.nelement()
        flops = weight_ops + bias_ops
        module_flops.append(flops)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hook_list.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hook_list.append(m.register_forward_hook(linear_hook))

    dummy_input = torch.rand(1, 3, input_size, input_size).cuda()
    #print(model)
    model(dummy_input)

    for hook in hook_list:
        hook.remove()
    return round(sum(module_flops) / 1e6, 2)


def calc_adaptive_model_flops(model, input_size, mul_add=False):
    hook_list = []
    module_flops = []

    def conv_hook(self, input, output):
        output_channels, output_height, output_width = output[0].size()
        bias_ops = 1 if self.bias is not None else 0
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        flops = (kernel_ops * (2 if mul_add else 1) + bias_ops) * output_channels * output_height * output_width
        module_flops.append(flops)

    def linear_hook(self, input, output):
        weight_ops = self.weight.nelement() * (2 if mul_add else 1)
        bias_ops = self.bias.nelement()
        flops = weight_ops + bias_ops
        module_flops.append(flops)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hook_list.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hook_list.append(m.register_forward_hook(linear_hook))

    dummy_input = torch.rand(1, 3, input_size, input_size).cuda()
    model=model.cuda()
    model(dummy_input)

    for hook in hook_list:
        hook.remove()
    return round(sum(module_flops) / 1e6, 2)


def calc_model_parameters(model):
    total_params = 0

    params = list(model.parameters())
    for param in params:
        cnt = 1
        for d in param.size():
            cnt *= d
        total_params += cnt

    return round(total_params / 1e6, 2)
