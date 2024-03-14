# -*- coding:utf-8  -*-

import numpy as np
import random
import torch
import torch.nn as nn

def make_divisible1(x, kernel_size=1,divisor=32, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8.

    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    
    if isinstance(kernel_size,tuple):
        kernel_size1=kernel_size[0]
    else:
        kernel_size1=kernel_size
            
    if min_value is None:
        min_value = divisor
     
    max_cin_next=4096//(kernel_size1*kernel_size1)
    if x>max_cin_next:
        x=max_cin_next
    
    new_v = max(min_value, int(x + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * x:
        new_v += divisor
    
    return int(new_v)

class AlphaLayer(nn.Module):
    def __init__(self, channels, min_width, max_width, offset, prob_type='exp'):
        super(AlphaLayer, self).__init__()

        assert prob_type in ['exp', 'sigmoid']
        self.prob_type = prob_type
        self.channels = channels

        ch_indice = self._get_ch_indice(min_width, max_width, offset, channels)
        self.min_ch = ch_indice[0]
        for i in range(1, len(ch_indice) - 1):
            assert ch_indice[i + 1] - ch_indice[i] == ch_indice[i] - ch_indice[i - 1]
        # no channels to pruned if num_groups == 0
        self.num_groups = len(ch_indice) - 1
        self.group_size = ch_indice[1] - ch_indice[0] if self.num_groups > 0 else 0
        assert self.group_size * self.num_groups + self.min_ch == channels

        self.register_buffer('alpha0', torch.ones(1))
        if self.num_groups > 0:
            self.alpha = nn.Parameter(torch.zeros(self.num_groups))
        else:
            self.alpha = None

    def get_condition_prob(self):
        if self.prob_type == 'exp':
            self.alpha.data.clamp_(min=0.)
            return torch.exp(-self.apha)
        elif self.prob_type == 'sigmoid':
            return torch.sigmoid(self.alpha)
        else:
            return NotImplementedError

    def get_marginal_prob(self):
        alpha = self.get_condition_prob()
        return torch.cumprod(alpha, dim=0)

    def expected_channel(self):
        if self.num_groups == 0:
            return  make_divisible1(self.min_ch)
        marginal_prob = self.get_marginal_prob()
        return make_divisible1(torch.sum(marginal_prob) * self.group_size + self.min_ch)

    def direct_sampling(self):
        """
        Direct sampling (DS): sampling independently by Markov process.
        """
        ##print('alpha_op:direct_sampling')
        if self.num_groups == 0:
            return make_divisible1(self.min_ch)
        prob = self.get_condition_prob().detach().cpu()

        pruned_ch = self.min_ch
        for i in range(self.num_groups):
            if random.uniform(0, 1) > prob[i]:
                break
            pruned_ch += self.group_size
        return make_divisible1(pruned_ch)

    def expected_sampling(self):
        """
        Expected sampling (ES): set the number of channels to be expected channels
        """
        expected = round(self.expected_channel() - 1e-4)
        candidate = [self.min_ch + self.group_size * i for i in range(self.num_groups + 1)]
        idx = np.argmin([abs(ch - expected) for ch in candidate])
        return candidate[idx], expected

    def forward(self, x):
        #print("alpha_op:")
        #print(x.shape)
        size_x = x.size()
        #min_ch=make_divisible1(self.min_ch)
        min_ch=self.min_ch
        if self.num_groups == 0 or size_x[1] == min_ch:
            return x
        
        #print('min_ch{}'.format(self.min_ch))
        prob = self.get_marginal_prob().view(self.num_groups, 1)
        tp_x = x.transpose(0, 1).contiguous()
        tp_group_x = tp_x[min_ch:]

        size_tp_group = tp_group_x.size()
         
        num_groups = size_tp_group[0] // self.group_size
        
        ##print('tp_group_x')
        #print(tp_group_x.shape)
        #print(num_groups)
        #print(self.group_size)
        if num_groups==0:
            return x
        n=tp_group_x.shape[0]*tp_group_x.shape[1]*tp_group_x.shape[2]*tp_group_x.shape[3]
        if  not (n%num_groups==0):
            return x
             
            
        tp_group_x = tp_group_x.view(num_groups, -1) * prob[:num_groups]
        tp_group_x = tp_group_x.view(size_tp_group)
        
        x = torch.cat([tp_x[:min_ch], tp_group_x]).transpose(0, 1).contiguous()
        #print('alhpha out:{}'.format(x.shape))
        return x

    @staticmethod
    def _get_ch_indice(min_width, max_width, width_offset, max_ch):
        ch_offset = int(width_offset * max_ch / max_width)
        num_offset = int((max_width - min_width) / width_offset + 1e-4)
        min_ch = max_ch - (ch_offset * num_offset)
        assert min_ch > 0

        indice = []
        for i in range(num_offset + 1):
            indice.append(min_ch + i * ch_offset)
        assert indice[0] == min_ch
        assert indice[-1] == max_ch

        return sorted(list(set(indice)))
