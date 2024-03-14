# -*- coding:utf-8  -*-

import math
import numpy as np
import os
import torch

import torch.nn as nn
import sys

sys.path.append("../../")
import tflite_quantization_PACT_weight_and_act as tflite

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def c_round(n):
    # 四舍五入
    return torch.where(n>0.0, torch.floor(n+0.5), torch.ceil(n-0.5))

def cal_M_n_v2(scale,  m_bits=12): 
    # 计算多分支结构的M和n：将一个浮点数分解成f=M*(2^n),限制M为m_bits
    print('cal_M_n_v2:{}'.format(scale))
    max_M=c_round(torch.tensor(2**m_bits-1) )
    #n = torch.floor(torch.log2(max_M / scale))
    print('max_M:{}'.format(max_M))
    n = torch.ceil(torch.log2(scale / max_M)) ####是负数
    print(n)

    # 取最小的n  
    #n = torch.tensor([c_round(torch.min(n))],device=scale.device)
    #print('n1:{}'.format(n))
    n_pow =  2 ** n
    print(n_pow)
    M = torch.floor(scale/n_pow) 
    return M, n

def hardware_round(n):
    # 硬件上的四舍五入 
    return torch.floor(n+0.5)


    
class AdaptiveBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, bottleneck_settings, stride=1, downsample=None):
        super(AdaptiveBasicBlock, self).__init__()
        conv1_in_ch, conv1_out_ch = bottleneck_settings['conv1']
        self.conv1 = conv3x3(conv1_in_ch, conv1_out_ch, stride)
        self.bn1 = nn.BatchNorm2d(conv1_out_ch)

        conv2_in_ch, conv2_out_ch = bottleneck_settings['conv2']
        self.conv2 = conv3x3(conv2_in_ch, conv2_out_ch)
        self.bn2 = nn.BatchNorm2d(conv2_out_ch)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample 
        '''
        #在训练时从外部注入
        if hasattr(m,'Mn_aware') and m.Mn_aware:
            m.register_buffer('block_M', torch.ones(1))
            m.register_buffer('block_n', torch.ones(0))
        '''
            
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        #######################################################
        clamp_range = tflite.c_round(2 ** (tflite.c_round(self.conv2.weight_bits) - 1) - 1)
        out = tflite.clamp(out,  min=-1*clamp_range, max=clamp_range)
        #######################################################

        return out
    def fuseforward(self, x):
        residual = x 

        out = self.conv1(x) 
        out = self.relu(out)

        out = self.conv2(out)

        # print("(x - residual).abs().mean():{}".format((x - residual).abs().mean()))
        
        #if self.downsample is not None:
            #residual = self.downsample(x) 
        if self.downsample is not None:
            
            if self.conv1.inference_type == "all_fp":
                x = tflite.Hardware_RoundFunction.apply(x / self.conv1.act_scale.detach()) 
                
                # ######### debug ###########
                # int_x = hardware_round(x*self.block_M*(2.0**self.block_n))
                # ######### debug ###########

                if not self.conv1.Mn_aware:
                    x = x * self.conv1.act_scale.detach()
                else:
                    scale= self.conv1.act_scale.detach() / self.downsample[0].act_scale.detach()
                    self.block_M,self.block_n=tflite.QuantizeMultiplier(scale)
                    #self.block_M,self.block_n= cal_M_n_v2(scale)
                    # print('2.0**self.block_n={}'.format(2.0**self.block_n))
                    x = x*self.block_M*(2.0**self.block_n)
                    x = tflite.Hardware_RoundFunction.apply(x)

                    # ###### 20240129新增，用于对齐训练和整型推理
                    x = tflite.clamp(x, min=-1.0*self.conv1.out_clamp_range, max=1.0*self.conv1.out_clamp_range)

                    x = x * self.downsample[0].act_scale.detach() 

                # ######### debug ###########
                # fp_x = hardware_round(x/self.downsample[0].act_scale.detach())
                # difference = (int_x - fp_x).abs().mean()
                # print("block difference:{}".format(difference))
                # ######### debug ###########
                
            # change the act scale for MN_AWARE
            #if self.downsample.inference_type == "full_int" and self.downsample.Mn_aware:
            elif self.downsample[0].inference_type == "full_int" and self.downsample[0].Mn_aware:
                ################## full interger inference ##################
                #scale=self.conv1.act_scale.detach() / self.downsample.act_scale.detach()
                # print('block_M:{},block_n:{}'.format(self.block_M,self.block_n))
                x = hardware_round(x*self.block_M*(2.0**self.block_n))

                ###### 20240129新增，用于对齐训练和整型推理
                x = tflite.clamp(x, min=-1 * self.downsample[0].out_clamp_range, max=self.downsample[0].out_clamp_range)
            
            # # debug
            # if self.conv1.inference_type == "all_fp": 
            #     print("residual intput min:{} max:{}".format(hardware_round(x.min()/self.downsample[0].act_scale.detach()), 
            #                                           hardware_round(x.max()/self.downsample[0].act_scale.detach())))
            # else:
            #     print("residual intput min:{} max:{}".format(x.min(), x.max()))

            residual = self.downsample(x)

            # # debug
            # if self.conv1.inference_type == "all_fp": 
            #     print("residual output min:{} max:{}".format(hardware_round(residual.min()/self.downsample[0].next_act_scale.detach()), 
            #                                           hardware_round(residual.max()/self.downsample[0].next_act_scale.detach())))
            # else:
            #     print("residual output min:{} max:{}".format(residual.min(), residual.max()))
            
        else: 
            # change the act scale for MN_AWARE
            
            if self.conv1.inference_type == "all_fp":
                x = tflite.Hardware_RoundFunction.apply(x / self.conv1.act_scale.detach()) 
                
                # ######### debug ###########
                # int_x = hardware_round(x*self.block_M*(2.0**self.block_n))#*scale
                # ######### debug ###########
                
                if not self.conv1.Mn_aware:
                    x=x* self.conv1.act_scale.detach()
                else:
                    # print('self.conv1.act_scale:{}'.format(self.conv1.act_scale))
                    # print('self.conv2.next_act_scale:{}'.format(self.conv2.next_act_scale))
                    scale=self.conv1.act_scale.detach()/ self.conv2.next_act_scale.detach()
                    self.block_M,self.block_n=tflite.QuantizeMultiplier(scale)
                    #self.block_M,self.block_n= cal_M_n_v2(scale)
                     
                    # print('block_M:{},block_n:{}'.format(self.block_M,self.block_n))
                    # print('2.0**self.block_n={}'.format(2.0**self.block_n))
                    x = x*self.block_M*(2.0**self.block_n)
                    x = tflite.Hardware_RoundFunction.apply(x)

                    # ###### 20240129新增，用于对齐训练和整型推理
                    x = tflite.clamp(x, min=-1.0*self.conv1.out_clamp_range, max=1.0*self.conv1.out_clamp_range)

                    x=x*self.conv2.next_act_scale.detach()

                # ######### debug ###########
                # fp_x = hardware_round(x/self.conv2.next_act_scale.detach())
                # difference = (int_x - fp_x).abs().mean()
                # print("block difference:{}".format(difference))
                # ######### debug ###########

            if self.conv1.inference_type == "full_int" and self.conv1.Mn_aware:
                ################## full interger inference ##################
                #scale=self.conv1.act_scale.detach() / self.conv2.next_act_scale.detach()
                # print('block_M:{},block_n:{}'.format(self.block_M,self.block_n))
                x = hardware_round(x*self.block_M*(2.0**self.block_n))#*scale
              
            residual = x

        out += residual
        out = self.relu(out)
        
        # debug
        # print("block M:{} n:{}".format(self.block_M,self.block_n))

        if self.conv1.inference_type == "full_int" and self.conv1.Mn_aware:
            ###########################
            N_overflow = tflite.is_overflow(out, self.conv2.weight_bits)
            print('resblock Overflow:{}'.format(N_overflow))
            ###########################
            #######################################################
            out = tflite.clamp(out, min=-1 * self.conv1.out_clamp_range, max=self.conv1.out_clamp_range)
        elif self.conv1.inference_type == "all_fp" and self.conv1.Mn_aware:
            # print("out: ", out.abs().max())
            # ###### 20240129新增，用于对齐训练和整型推理
            out = tflite.Hardware_RoundFunction.apply(out/self.conv2.next_act_scale.detach())
            out = tflite.clamp(out, min=-1.0 * self.conv1.out_clamp_range, max=1.0 *self.conv1.out_clamp_range)
            out = out * self.conv2.next_act_scale.detach()
          
        
        return out


class AdaptiveBottleneck(nn.Module):
    expansion = 4

    def __init__(self, bottleneck_settings, stride=1, downsample=None):
        super(AdaptiveBottleneck, self).__init__()
        conv1_in_ch, conv1_out_ch = bottleneck_settings['conv1']
        self.conv1 = nn.Conv2d(conv1_in_ch, conv1_out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_out_ch)

        conv2_in_ch, conv2_out_ch = bottleneck_settings['conv2']
        self.conv2 = nn.Conv2d(conv2_in_ch, conv2_out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv2_out_ch)

        conv3_in_ch, conv3_out_ch = bottleneck_settings['conv3']
        self.conv3 = nn.Conv2d(conv3_in_ch, conv3_out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv3_out_ch)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        

        return out


class AdaptiveResNet(nn.Module):
    def __init__(self, ch_cfg, block, layers, num_classes=1000, input_size=224):
        super(AdaptiveResNet, self).__init__()

        channels = np.load(os.path.join(ch_cfg, 'sample.npy'), allow_pickle=True).item()
        self.inplanes = 64 
        #self.conv1 = nn.Conv2d(3, channels['conv1'], kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], 1, channels['layer1'])
        self.layer2 = self._make_layer(block, 128, layers[1], 2, channels['layer2'])
        self.layer3 = self._make_layer(block, 256, layers[2], 2, channels['layer3'])
        #self.layer4 = self._make_layer(block, 512, layers[3], 2, channels['layer4'])
        
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=1)
        #self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(128)
        #self.relu2 = nn.ReLU(inplace=True)
        
        self.layer4 = self._make_layer(block, 256, layers[3], 2, channels['layer4'])
        
        #self.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, padding=1)
        #self.fc = nn.Linear(channels['fc'], num_classes)
        self.fc = nn.Conv2d(352*3*3, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.maxpool2(x)  ##########

        x = self.layer4(x)

        x = self.maxpool3(x)  #########
        # x = x.view(x.size(0), -1)
        # print('fc input0:.............')
        # print(x.shape)
        x = x.view(x.size(0), -1, 1, 1)
        # print('fc input:.............')
        # print(x.shape)
        # print(self.fc.weight.shape)

        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x
    def fuseforward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.maxpool2(x)  #########

        x = self.layer4(x)
        # print('avgpool input size:{}'.format(x.shape))
        # x = self.avgpool(x)
        x = self.maxpool3(x)  #########

        # print(x)
        # x = x.view(x.size(0), -1)
        x = x.view(x.size(0), -1, 1, 1)
        # print('fc input:')
        # print(x.shape)
        # print(x)
        # print(self.fc.int_weight)

        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, bottleneck_settings):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            in_ch, _ = bottleneck_settings['0']['conv1']
            if 'conv3' in bottleneck_settings['0'].keys():
                _, out_ch = bottleneck_settings['0']['conv3']
            else:
                # basic block
                _, out_ch = bottleneck_settings['0']['conv2']

            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        layers = [block(bottleneck_settings['0'], stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(bottleneck_settings[str(i)]))

        return nn.Sequential(*layers)


def adaptive_res18(ch_cfg, num_classes=1000, input_size=224):
    return AdaptiveResNet(ch_cfg, AdaptiveBasicBlock, [2, 2, 2, 2], num_classes, input_size)


def adaptive_res50(ch_cfg, num_classes=1000, input_size=224):
    return AdaptiveResNet(ch_cfg, AdaptiveBottleneck, [3, 4, 6, 3], num_classes, input_size)
