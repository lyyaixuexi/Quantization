import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
import torch.distributed as dist
import copy
import math
import numpy as np
from overflow_utils import *  

DEBUG_Q=0
g_conv_id=0

from  rwfile import *
def c_round(n):
    # 四舍五入
    return torch.where(n>0.0, torch.floor(n+0.5), torch.ceil(n-0.5))

def hardware_round(n):
    # 硬件上的四舍五入
    '''n = n.double()
    n = torch.where(n>0.0, torch.floor(n+0.5), torch.ceil(n-0.499999999999))
    return n.float()
    '''
    return torch.floor(n+0.5)

def  QuantizeMultiplier(double_multiplier,m_bits=12):
    # print('multiplier:{}'.format(double_multiplier))
    quantized_multiplier = 1
    shift = 0
    if double_multiplier == 0:  
        quantized_multiplier = 0
        shift = 0
        return quantized_multiplier,shift
    
    q,shift= torch.frexp(double_multiplier)
    # print(q)
    #q_fixed =c_round(q * c_round(torch.tensor(2**m_bits)))
    #q_fixed=c_round(q)
    #print(q_fixed)
    #shift-=m_bits
    q_fixed=q
    
    
    max_M=c_round(torch.tensor(2.0**m_bits-1))
    # print('max_M:{}'.format(max_M))
    
    while q_fixed<max_M: #让q_fixed接近max_M,使shift尽可能小
        #q_fixed=c_round(q_fixed*2)
        shift=shift-1 
        #print('shift={}'.format(shift))
        #print((2.0**shift))
        m=double_multiplier/(2.0**shift) ###################
        #print('m:{}'.format(m))
        q_fixed=torch.tensor(m)
        q_fixed=c_round(q_fixed)    
        #print('good0......')
        #print(q_fixed)
        #print(shift)
  #if (q_fixed == (1ll << 31)) {
    while q_fixed>max_M:
        #q_fixed=c_round(q_fixed/2)
        shift=shift+1
        q_fixed=torch.tensor(double_multiplier/(2.0**shift))
        q_fixed=c_round(q_fixed)
        #print('good...')
        #print(q_fixed)
        #print(shift)
    
   
    if shift < -31:
        shift = 0
        q_fixed= 1
        print('shift<-31....................................')
  
    quantized_multiplier =q_fixed
    # print('scale={},M*2^^n={}'.format(double_multiplier,q_fixed*(2.0**shift)))
    return torch.tensor(quantized_multiplier),torch.tensor(shift)
 

def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)


def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)

def reshape_to_bias(input):
    return input.reshape(-1)
def fuse_conv_and_bn0(conv, bn):
    # 融合Bn层和卷积层的参数，去除Bn层
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = copy.deepcopy(conv)
    if conv.bias is None:
        fusedconv.bias = nn.Parameter(torch.zeros(fusedconv.out_channels, device=conv.weight.device))

    fusedconv = fusedconv.requires_grad_(False)

    # prepare filters
    w_conv = conv.weight.detach().clone().view(conv.out_channels, -1).requires_grad_(False)
    w_bn = torch.diag(bn.weight.detach().clone().div(torch.sqrt(bn.eps + bn.running_var))).requires_grad_(False)
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean.detach()).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn.detach(), b_conv.detach().reshape(-1, 1)).reshape(-1) + b_bn.detach())
    fusedconv.fuseBN = True

    return fusedconv
def fuse_conv_and_bn(conv, bn):
    #在这里只是去掉了BN层，参数的fuse放在Conv2d_auantization
    
    with torch.no_grad():
        '''
        conv.eps = 1e-5
        conv.momentum = 0.01
        conv.gamma = nn.Parameter(torch.Tensor(conv.out_channels))
        conv.beta = nn.Parameter(torch.Tensor(conv.out_channels))
        conv.register_buffer('running_mean', torch.zeros(conv.out_channels))
        conv.register_buffer('running_var', torch.ones(conv.out_channels))
        '''
        conv.running_mean.copy_(bn.running_mean)
        conv.running_var.copy_(bn.running_var)
        conv.gamma.copy_(bn.weight)
        conv.beta.copy_(bn.bias)
        conv.fuseBN=True

    return conv


def clamp(x, min, max):
    # 约束取值范围
    x = torch.where(x<max, x, max)
    x = torch.where(x>min, x, min)
    return x

# Uniform quantization
def uniform_symmetric_quantizer_per_tensor(x, num_levels, bits=8, bias_scale=None, cal_max=None, alpha=None):
    # 对整个tensor进行对称均匀量化
    # per_tensor, signed=True
    # calculate scale
    if bias_scale is not None:
        # quantize the bias
        scale = bias_scale

        # quantize
        x_int = Hardware_RoundFunction.apply(x / scale)

        # clamp
        num_levels = int(num_levels)
        x_int = torch.clamp(x_int, -1 * (num_levels / 2 - 1), (num_levels / 2 - 1))

        # dequantize
        x_dequant = x_int * scale

        return x_dequant

    else:
        if cal_max == None:
            maxv = torch.max(torch.abs(x))
        else:
            maxv = cal_max
            
        cal_max=maxv
        
        minv = - maxv
        scale = (maxv - minv+1e-5) / (num_levels - 2)
       # print('scale...........')
       # print(scale)

        if alpha is not None:
            scale = scale * alpha

        # clamp
        x = clamp(x, min=minv, max=maxv)
        # quantize
        x_int = Hardware_RoundFunction.apply(x / scale)

        # clamp
        num_levels = int(num_levels)
        x_int = torch.clamp(x_int, -1 * (num_levels / 2 - 1), (num_levels / 2 - 1))

        # dequantize
        x_dequant = x_int * scale

        return x_dequant, scale,cal_max

def uniform_symmetric_quantizer_per_channel(x, num_levels, bits=8, bias_scale=None, cal_max=None, alpha=None, bias=None, acc_clamp_range=None, act_scale=None):
    # 对每个通道分别进行对称均匀量化
    # per_channel, signed=True
    # calculate scale
    if bias_scale is not None:
        # quantize the bias
        scale = bias_scale

        # quantize
        x_int = RoundFunction.apply(x / scale)

        # clamp
        num_levels = int(num_levels)
        x_int = torch.clamp(x_int, -1*(num_levels/2-1), (num_levels/2-1))

        # dequantize
        x_dequant = x_int * scale

        return x_dequant

    else:
        if cal_max is not None:
            maxv = cal_max

        else:
            c_out, c_in, k, w = x.shape
            x_reshape = x.reshape(c_out, -1)
            maxv, _ = torch.max(torch.abs(x_reshape), 1)   # maxv shape = c_out*1
            
         
        
        minv = - maxv
        scale = (maxv - minv+1e-5) / (num_levels - 2)   # scale shape = c_out*1

        if bias is not None:
            temp = acc_clamp_range / 2 * act_scale.detach() / bias.detach().abs()
            min_scale = 1 / temp.floor()
            # print("visual temp and min_scale")
            # print(temp)
            # print(min_scale)
            scale = torch.where(scale > min_scale, scale, min_scale)

        if alpha is not None:
            scale = scale * alpha

        scale_expand = scale.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)

        maxv = maxv.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
        minv = minv.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
        x = clamp(x, min=minv, max=maxv) ################

        # quantize
        x_int = RoundFunction.apply(x / scale_expand)

        # clamp
        num_levels = int(num_levels)
        x_int = torch.clamp(x_int, -1 * (num_levels / 2 - 1), (num_levels / 2 - 1))

        # dequantize
        x_dequant = x_int * scale_expand

        return x_dequant, scale 

class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return c_round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Hardware_RoundFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return hardware_round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def get_scale_approximation_shift_bits(fp32_scale, mult_bits):

    shift_bits = torch.floor(torch.log2((2 ** mult_bits - 1) / fp32_scale))
    # print('shift_bits.shape={},\nshift_bits={}'.format(shift_bits.shape, shift_bits))
    shift_bits = torch.min(mult_bits, shift_bits)
    # print('shift_bits.shape={},\nshift_bits={}'.format(shift_bits.shape, shift_bits))
    return shift_bits


def get_scale_approximation_mult(fp32_scale, shift_bits):
    return torch.floor(fp32_scale * (2 ** shift_bits))


def get_scale_approximation(fp32_scale, mult_bits):
    shift_bits = get_scale_approximation_shift_bits(fp32_scale, mult_bits)
    # print('shift_bits={}'.format(shift_bits))
    multiplier = get_scale_approximation_mult(fp32_scale, shift_bits)
    # print('fp32_scale={}'.format(fp32_scale))
    # print('shift_bits={}'.format(shift_bits))
    # print('multiplier={}'.format(multiplier))
    scale_int = multiplier / (2 ** shift_bits)
    # print('scale_int={}'.format(scale_int))
    return scale_int


class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output
    
class Gather(Function):
    @staticmethod
    def forward(ctx, input):
        total_input=torch.zeros(dist.get_world_size(), input.size()[0], device=input.device)
        dist.all_gather(list(total_input.chunk(dist.get_world_size(), dim=0)), input.data)
        total_input.requires_grad=True
        return total_input

    @staticmethod
    def backward(ctx, grad_output):
        grad_x=None
        if grad_output is not None:
            grad_output.detach_()
            x_grad=torch.zeros(grad_output.size()[1], device=grad_output.device)
            dist.reduce_scatter(x_grad, list(grad_output.chunk(dist.get_world_size(), dim=0)))
            grad_x = x_grad
        return grad_x
    
def cal_M_n(act_scale, weight_scale, next_act_scale, m_bits, max_M):

    # 计算卷积后处理的M和n
    M_fp = act_scale * weight_scale / next_act_scale
    n = torch.floor(torch.log2(max_M / M_fp))

    # 取最小的n

    # test
    # print("n min:{} max:{}".format(n.min(), n.max()))

    n = torch.tensor([c_round(torch.min(n))],device=M_fp.device)
    n_pow = c_round(2 ** n)
    M = torch.floor(M_fp * n_pow)
    # M_fp = M / n_pow

    # if M.min() == 0:
    #     print("remove some channels!!!!!!")
    #
    # if M_fp.min() == 0:
    #     print("remove some channels!!!!!!")

    if M.max() > max_M:
        print("############################ M overflow happen ###################")
        print("M.max():".format(M.max()))

    return M, n, n_pow

def is_overflow(output,bits): 
    global g_conv_id
    max = 2 ** (bits - 1) - 1
    min =-max
    N_overflow=0
    N,C,H,W=output.shape
    nover=(output>max)
    nlower=(output<min)
    N_overflow=nover.sum()+nlower.sum()
    if N_overflow>0:
        print('layer_id:{} Overflow:{}'.format(g_conv_id, N_overflow))
        # print('Overflow:{}......................,up_overflow={},low_overflow={}'.format(N_overflow,nover,nlower))
    
    return N_overflow
    
########部分量化：将网络分成两部分：一部分量化，一部分不量化##############################
b_split_2part=False  #####是否采用部分量化
int_start_conv=19#17 #从哪层开始量化
N_layers=21 #20Convs+1Conv(FC转化而来) #整个网络的卷积数 
g_conv_id=0  #Conv ID: 0~N_layers-1  (递增时要使用求余符号)
##########################################

class Conv2d_quantization(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 act_bits=8, weight_bits=8, m_bits=12, bias_bits=16, inference_type="all_fp",
                 fuseBN=False, classify_layer=False, Mn_aware=False, power_of_two_scale=False):
        super(Conv2d_quantization, self).__init__(in_channels, out_channels, kernel_size,
                                               stride, padding, dilation, groups, bias)

        # 激活值的bit数
        self.register_buffer('act_bits', torch.Tensor([act_bits]))

        # 权重的bit数
        self.register_buffer('weight_bits', torch.Tensor([weight_bits]))

         # m的比特数
        self.register_buffer('m_bits', torch.Tensor([m_bits]))

        # 累加器的比特数
        self.register_buffer('bias_bits', torch.Tensor([bias_bits]))

        # 推理类型，分为两种：all_fp 和 full_int, 其中all_fp用于训练，full_int则是仿真的整数推理
        self.inference_type = inference_type

        # 是否融合BN到卷积层
        self.fuseBN = fuseBN
        
        # 是否为网络的输出层
        self.classify_layer = classify_layer

        # 是否模拟卷积层后处理
        self.Mn_aware = Mn_aware

        # 激活值scale，可通过act_max计算得到
        #self.register_buffer('act_scale', torch.ones(1)/127.5)
        self.register_buffer('act_scale', torch.ones(1))
        
        # 权重scale，可通过weight的最大值计算得到
        self.register_buffer('weight_scale', torch.zeros(out_channels))

        # bias_scale, 直接通过act_scale和weight_scale计算得出
        self.register_buffer('bias_scale', torch.zeros(out_channels))

        # 对量化后取值范围的约束系数，会在训练过程自动更新
        self.register_buffer('alpha', torch.Tensor([1.0]))

        # 用于存储下一层的激活值的scle
        self.register_buffer('next_act_scale', torch.Tensor([1.0]))

        # PACT activation parameter
        # 激活值的浮点数范围，用于约束激活值的大小和决定act_scale
        self.act_max = nn.Parameter(torch.ones(1)*6)#####6)

        self.register_buffer('act_num_levels', c_round(2 ** self.act_bits))
        self.register_buffer('weight_num_levels', c_round(2 ** self.weight_bits))
        self.register_buffer('bias_num_levels', c_round(2 ** self.bias_bits))

        if self.Mn_aware:
            # 用于卷积后处理的M和n，可通过act_scale，weight_scale和next_act_scale计算得到
            self.register_buffer('M', torch.zeros(out_channels))
            self.register_buffer('n', torch.zeros(1))
            if self.classify_layer:
                # self.register_buffer('pred_max', torch.ones(1)*20)
                # 输出的取值范围，可学习
                if self.weight_bits>4: 
                     self.pred_max = nn.Parameter(torch.ones(1)*20)  
                else:#4bit
                    self.pred_max = nn.Parameter(torch.ones(1)*60) #20

        self.register_buffer('acc_clamp_range', c_round(2 ** (c_round(self.bias_bits) - 1) - 1))
        self.register_buffer('out_clamp_range',  c_round(2 ** (c_round(self.weight_bits) - 1) - 1))
        self.register_buffer('max_M', c_round(2 ** self.m_bits - 1))
        
        if self.fuseBN:
            self.eps = 1e-5
            self.momentum = 0.01
            self.gamma = nn.Parameter(torch.Tensor(out_channels))
            self.beta = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer('running_mean', torch.zeros(out_channels))
            self.register_buffer('running_var', torch.ones(out_channels))

            torch.nn.init.uniform_(self.gamma)
            torch.nn.init.zeros_(self.beta)
        
        self.power_of_two_scale = power_of_two_scale

    def forward(self, input): 
        global nimg
        global g_conv_id
        global b_split_2part
        global int_start_conv
        
         
        #if DEBUG_Q:
            #print("weight bits{}".format(self.weight_bits))
            #print("bias bits{}".format(self.bias_bits))
            #print("act bits{}".format(self.act_bits))
        # 训练的时候使用这个模式，用于模拟量化的过程
        #print('g_conv_id={}..............'.format(g_conv_id))
        device=input.get_device()
        
        # 训练的时候使用这个模式，用于模拟量化的过程
        if  (self.inference_type == "all_fp" and (not self.fuseBN)) or (self.inference_type == "full_int"): 
            if b_split_2part: #####stage1
                    if g_conv_id<int_start_conv:
                        #print('float forward...............')
                        output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding,
                                self.dilation, self.groups)
                        if DEBUG_Q :
                            openoutfile()
                            writeoutfile("Conv{}_weight:\n".format(g_conv_id))
                            Cout= self.weight.shape[0]
                            Cin = self.weight.shape[1]
                            K1 =self.weight.shape[2]
                            K2 = self.weight.shape[3]
                            writeoutfile("weight:Cout={},Cin={},K1={},K2={}\n".format(Cout, Cin, K1,K2))
                            for co in range(Cout):
                                for k1 in range(K1):
                                    for k2 in range(K2):
                                        for ci in range(Cin):
                                            writeoutfile("{:.4f},".format(float(self.weight[co][ci][k1][k2])))
                                        writeoutfile("\n")
                            
                            writeoutfile("Conv bias:\n")
                            for co in range(Cout):
                                   writeoutfile("{:.4f},".format(float(self.bias[co])))
                            writeoutfile("\n")
                            closeoutfile()
                
                        if DEBUG_Q:
                            openoutfile()
                            writeoutfile("Conv{}_input before preprocess:\n".format(g_conv_id))
                            N= input.shape[0]
                            C= input.shape[1]
                            H=input.shape[2]
                            W= input.shape[3]
                            writeoutfile("input:Cout={},Cin={},K1={},K2={}\n".format(N, C, H,W))
                            for n in range(N):
                                for h in range(H):
                                    for w in range(W):
                                        for c in range(C):
                                            writeoutfile("{:.4f},".format(float(input[n][c][h][w])))
                                        writeoutfile("\n")
                            closeoutfile() 
                        if DEBUG_Q:
                            openoutfile()

                            N = output.shape[0]
                            C = output.shape[1]
                            H = output.shape[2]
                            W =output.shape[3]
                            writeoutfile("Conv{},N={},C={},H={},W={}\n".format((g_conv_id),N, C, H, W))
                            print("Conv{},size: N={},C={},H={},W={}".format((g_conv_id),N, C, H, W))
                    

                            for h in range(H):
                                for w in range(W):
                                    for c in range(C):
                                        writeoutfile("{:.4f},".format(float(output[0][c][h][w])))
                                    writeoutfile("\n")
                            closeoutfile() 
                        g_conv_id=(g_conv_id+1)%N_layers
                        return output
        
        if self.inference_type == "all_fp": 
            
            # ########## temp code for debug ###########
            # # print("debug full int==>act_scale:{}  M:{} n:{}".format(self.act_scale,self.M, self.n))
            # int_input=hardware_round(input/self.act_scale)
            # clamp_range = c_round(2 ** (c_round(self.weight_bits) - 1) - 1)
            # int_input = clamp(int_input, min=-1 * clamp_range, max=clamp_range)

            # int_output = F.conv2d(int_input, self.int_weight, self.int_bias, self.stride, self.padding,
            #                                             self.dilation, self.groups)
            
            # #############################
            # is_overflow(int_output, self.bias_bits)
            # ###########################
            # # # 卷积结果后处理
            # M_expand = self.M.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(int_output)
            # pow_n_expand = c_round(torch.pow(2, c_round(self.n)))
            # int_output = hardware_round(int_output * M_expand / pow_n_expand)

            # int_output = clamp(int_output,  min=-1*self.out_clamp_range, max=self.out_clamp_range)
            
            # ########## temp code for debug ###########

            # float inference
            if self.fuseBN:
                ########stage2: fuseBN ################################################ 
                if self.training:
                    output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding,
                                      self.dilation, self.groups)

                    # 更新BN统计参数（batch和running）
                    dims = [dim for dim in range(4) if dim != 1]
                    local_batch_mean = torch.mean(output, dim=dims)
                    local_batch_var = torch.var(output, dim=dims)
                    total_batch_mean=Gather.apply(local_batch_mean)
                    total_batch_var=Gather.apply(local_batch_var)
                    batch_mean=torch.mean(total_batch_mean, dim=0)
                    batch_var=torch.mean(total_batch_var, dim=0)

                    with torch.no_grad():
                        self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                        self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)

                    # BN融合
                    if self.bias is not None:
                        bias = reshape_to_bias(
                            self.beta + (self.bias - batch_mean) * (self.gamma / torch.sqrt(batch_var + self.eps)))
                    else:
                        bias = reshape_to_bias(self.beta - batch_mean * (
                            self.gamma / torch.sqrt(batch_var + self.eps)))  # b融batch
                    weight = self.weight * reshape_to_weight(
                        self.gamma / torch.sqrt(self.running_var + self.eps))  # w融running
                else:
                    #stage2 inference##############################
                    
                    # print("testing")
                    # BN融合
                    if self.bias is not None:
                        bias = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (self.gamma / torch.sqrt(self.running_var + self.eps)))
                    else:
                        bias = reshape_to_bias(self.beta - self.running_mean * (self.gamma / torch.sqrt(self.running_var + self.eps)))  # b融running
                    weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))  # w融running
                #######################################
                if b_split_2part: ######stage2
                    if g_conv_id<int_start_conv:  
                        output = F.conv2d(input, weight, bias, self.stride, self.padding,self.dilation, self.groups) 
                        g_conv_id=(g_conv_id+1)%N_layers
                        return output
                ####################################### 
                
                # 对卷积层的输入进行模拟量化
                #quantized_input, self.act_scale,self.act_max = uniform_symmetric_quantizer_per_tensor(input, num_levels=self.act_num_levels, bits=self.act_bits, cal_max=self.act_max, alpha=self.alpha) ###为适配硬件而去掉alpha
                quantized_input, self.act_scale,self.act_max = uniform_symmetric_quantizer_per_tensor(input, num_levels=self.act_num_levels, bits=self.act_bits, cal_max=nn.Parameter(self.act_max.abs()), alpha=None)#输入要截断到固定quantization_bits

                # 对卷积层的权重进行模拟量化
                quantized_weight, self.weight_scale = uniform_symmetric_quantizer_per_channel\
                    (weight, num_levels=self.weight_num_levels, bits=self.weight_bits, alpha=self.alpha, bias=bias,
                        acc_clamp_range=self.acc_clamp_range, act_scale=self.act_scale)
                
                # 对卷积层的bias进行模拟量化
                # quantization self.bias
                #if self.bias is not None:#####self.bais maybe None,but fuseBN has 'bias'
                if bias is not None:###########################################################
                    self.bias_scale = self.act_scale * self.weight_scale
                    quantized_bias = uniform_symmetric_quantizer_per_channel\
                        (bias, num_levels=self.bias_num_levels, bits=self.bias_bits, bias_scale=self.bias_scale)
                else:
                    # not quantization self.bias
                    quantized_bias = None

                # 使用模拟量化后的输入和参数计算卷积结果
                if self.training:
                    output = F.conv2d(quantized_input, quantized_weight, None, self.stride, self.padding, self.dilation, self.groups) 
                    output = output * reshape_to_activation(
                        torch.sqrt(self.running_var + self.eps) / torch.sqrt(batch_var + self.eps))
                    if bias is not None:####################################################
                        output = output + reshape_to_activation(quantized_bias)

                else:
                    output = F.conv2d(quantized_input, quantized_weight, quantized_bias, self.stride, self.padding,
                                      self.dilation, self.groups)

            else:
                #### stage1: simulated quantization (not fuseBN) ##################################
                ############################################################3
                # 对卷积层的输入进行模拟量化
                ####################
                #input1= input.detach() *self.act_scale.detach()  #########
                ###################
                #print('input:conv{}......................'.format(g_conv_id))
                #print(input)
                
                #quantized_input, self.act_scale , calc_max= uniform_symmetric_quantizer_per_tensor (input, num_levels=self.act_num_levels, bits=self.act_bits, cal_max=self.act_max, alpha=self.alpha)###为适配硬件而去掉alpha
                quantized_input, self.act_scale , calc_max= uniform_symmetric_quantizer_per_tensor (input, num_levels=self.act_num_levels, bits=self.act_bits, cal_max=nn.Parameter(self.act_max.abs()), alpha=None)
                #print('quantized_input:conv{}..................................'.format(g_conv_id))
                #print(quantized_input)
                #quantized_input, self.act_scale, calc_max= uniform_symmetric_quantizer_per_tensor\
                    #(input, num_levels=self.act_num_levels, bits=self.act_bits,  alpha=self.alpha)
                #self.act_max.data=calc_max
                
                # 对卷积层的权重、bias进行模拟量化
                if self.bias is not None:
                    quantized_weight, self.weight_scale = uniform_symmetric_quantizer_per_channel\
                        (self.weight, num_levels=self.weight_num_levels, bits=self.weight_bits, alpha=self.alpha,
                         bias=self.bias, acc_clamp_range=self.acc_clamp_range, act_scale=self.act_scale)
                    # quantization self.bias
                    self.bias_scale = self.act_scale * self.weight_scale
                    quantized_bias = uniform_symmetric_quantizer_per_channel \
                        (self.bias, num_levels=self.bias_num_levels, bits=self.bias_bits, bias_scale=self.bias_scale)
                else:
                    quantized_weight, self.weight_scale = uniform_symmetric_quantizer_per_channel\
                        (self.weight, num_levels=self.weight_num_levels, bits=self.weight_bits, alpha=self.alpha)
                    # not quantization self.bias
                    quantized_bias = None

                # 使用模拟量化后的输入和参数计算卷积结果
                output = F.conv2d(quantized_input, quantized_weight, quantized_bias, self.stride, self.padding,
                                self.dilation, self.groups)

                if DEBUG_Q:
                    openoutfile()
                    writeoutfile("Conv{}_weight:\n".format(g_conv_id))
                    Cout= quantized_weight.shape[0]
                    Cin = quantized_weight.shape[1]
                    K1 =quantized_weight.shape[2]
                    K2 = quantized_weight.shape[3]
                    writeoutfile("weight:Cout={},Cin={},K1={},K2={}\n".format(Cout, Cin, K1,K2))
                    for co in range(Cout):
                        for k1 in range(K1):
                            for k2 in range(K2):
                                for ci in range(Cin):
                                    writeoutfile("{},".format(float(quantized_weight[co][ci][k1][k2])))
                                writeoutfile("\n")
                    closeoutfile()

                if DEBUG_Q:
                    openoutfile()
                    writeoutfile("Conv{}_input before preprocess:\n".format(g_conv_id))
                    N= quantized_input.shape[0]
                    C= quantized_input.shape[1]
                    H=quantized_input.shape[2]
                    W= quantized_input.shape[3]
                    writeoutfile("input:Cout={},Cin={},K1={},K2={}\n".format(N, C, H,W))
                    for n in range(N):
                        for h in range(H):
                            for w in range(W):
                                for c in range(C):
                                    writeoutfile("{},".format(float(quantized_input[n][c][h][w])))
                                writeoutfile("\n")
                    closeoutfile()
                if DEBUG_Q:
                    openoutfile()

                    N = output.shape[0]
                    C = output.shape[1]
                    H = output.shape[2]
                    W = output.shape[3]
                    writeoutfile("Conv{},N={},C={},H={},W={}\n".format((g_conv_id),N, C, H, W))
                    # print("Conv{},size: N={},C={},H={},W={}".format((g_conv_id),N, C, H, W))
                    # print('out_clamp_range={}'.format(self.out_clamp_range))

                    for h in range(H):
                        for w in range(W):
                            for c in range(C):
                                writeoutfile("{},".format(int(output[0][c][h][w])))
                            writeoutfile("\n")
                    closeoutfile()

            self.bias_scale = self.act_scale * self.weight_scale
             
            # 对输出层进行卷积后处理仿真训练
            if self.Mn_aware: 
                ####### stage3: S1*S2/S3----->M/n (post-proccessing:  adapted for the bits for accumulator)
                ###########################################################################
                if self.classify_layer:
                    # convert fp32 to int16
                    bias_scale_expand = self.bias_scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(output).detach()
                    output = RoundFunction.apply(output / bias_scale_expand)
        
                    #############################
                    is_overflow(output, self.bias_bits)
                    ###########################

                    #output = clamp(output, min=-1 * self.acc_clamp_range, max=self.acc_clamp_range) #为适配硬件而去掉

                    # convert int16 to int8
                    pred_scale = self.pred_max / self.out_clamp_range
                    
                    self.M, self.n, n_pow = cal_M_n(self.act_scale, self.weight_scale, pred_scale, self.m_bits, self.max_M)
                    
                    M_expand = self.M.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(output).detach()
                    output = Hardware_RoundFunction.apply(output * M_expand / n_pow)

                    output = clamp(output, min=-1.0 * self.out_clamp_range, max=1.0*self.out_clamp_range)

                    # convert int8 to fp32
                    output = output * pred_scale
                    #print(torch.max(output))
                
                # 对融合Bn后的卷积层进行卷积后处理仿真训练
                else:
                    # convert fp32 to int16
                    bias_scale_expand = self.bias_scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(output).detach()
                    output = RoundFunction.apply(output / bias_scale_expand)

                    #############################
                    is_overflow(output, self.bias_bits)
                    ###########################

                    #output = clamp(output,  min=-1*self.acc_clamp_range, max=self.acc_clamp_range) #为适配硬件而去掉

                    # convert int16 to int8
                    self.M, self.n, n_pow = cal_M_n(self.act_scale, self.weight_scale, self.next_act_scale, self.m_bits, self.max_M)

                    M_expand = self.M.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(output).detach()
                    # output = hardware_round(output * M_fp_expand)
                     
                    output = Hardware_RoundFunction.apply(output * M_expand / n_pow)

                    # print("##############################################")
                    # print(output.dtype)
                    # print((-1.0*self.out_clamp_range).dtype)
                    # print((1.0*self.out_clamp_range).dtype)

                    output = clamp(output,  min=-1.0*self.out_clamp_range, max=1.0*self.out_clamp_range)

                    # convert int8 to fp32
                     
                    output = output * self.next_act_scale


            # ########## temp code for debug ###########
            # if self.classify_layer:
            #     # convert output from interger to float 
            #     fp_output = hardware_round(output/pred_scale)
            # else:
            #     fp_output = hardware_round(output/self.next_act_scale)

            # difference_1 = (int_output_1 - fp_output_1).abs().mean()
            # difference_2 = (int_output_2 - fp_output_2).abs().mean()
            # difference = (int_output - fp_output).abs().mean()
            # # print("debug all_fp==>act_scale:{} M:{} n:{}".format(self.act_scale, self.M, self.n))
    
            # print("g_conv_id:{} difference_1:{} difference_2:{} difference:{}".format(g_conv_id, difference_1, difference_2, difference))
            # ########## temp code for debug ###########

            g_conv_id=(g_conv_id+1)%N_layers
            if DEBUG_Q:
                print(output)
            return output
      
            

        if self.inference_type == "full_int":
            if DEBUG_Q :
                openoutfile()
                writeoutfile("Conv{}_weight:\n".format(g_conv_id))
                Cout= self.int_weight.shape[0]
                Cin = self.int_weight.shape[1]
                K1 =self.int_weight.shape[2]
                K2 = self.int_weight.shape[3]
                writeoutfile("weight:Cout={},Cin={},K1={},K2={}\n".format(Cout, Cin, K1,K2))
                for co in range(Cout):
                    for k1 in range(K1):
                        for k2 in range(K2):
                            for ci in range(Cin):
                                writeoutfile("{},".format(int(self.int_weight[co][ci][k1][k2])))
                            writeoutfile("\n")
                            
                writeoutfile("Conv bias:\n")
                for co in range(Cout):
                       writeoutfile("{:.4f},".format(float(self.int_bias[co])))
                writeoutfile("\n")
                closeoutfile()
            
            if DEBUG_Q:
                openoutfile()
                writeoutfile("Conv{}_input before hardware_round:\n".format(g_conv_id))
                N= input.shape[0]
                C= input.shape[1]
                H=input.shape[2]
                W= input.shape[3]
                writeoutfile("input:Cout={},Cin={},K1={},K2={}\n".format(N, C, H,W))
                for n in range(N):
                    for h in range(H):
                        for w in range(W):
                            for c in range(C):
                                writeoutfile("{:.5f},".format(float(input[n][c][h][w])))
                            writeoutfile("\n")
                closeoutfile()
                
            ###############################################
            
            #之后放在网络结构中处理
            if b_split_2part:
                if (g_conv_id==int_start_conv or g_conv_id==int_start_conv+1):#量化3层
            #if b_split_2part and (g_conv_id==int_start_conv):
                    #quantized_input, _,_= uniform_symmetric_quantizer_per_tensor(input, num_levels=self.act_num_levels, bits=self.act_bits, cal_max=self.act_max, alpha=self.alpha)
                    int_input=hardware_round(input/self.act_scale)
                    # print('g_conv_id={},act_scale={}'.format(g_conv_id,self.act_scale))
            else :
                if g_conv_id==0:
                    int_input=hardware_round(input/self.act_scale)
                    # 20240129新增的，对模型输入的clamp
                    # int_input = clamp(int_input, c_round(-1 * self.act_max / self.act_scale), c_round(self.act_max / self.act_scale))  ###为适配硬件而注释掉
                    #print("g_conv_id==0..................")
                    #print(int_input.tolist())
                else:
                    int_input = hardware_round(input)  
            
            # int_input = hardware_round(input) 
           ###############################################   
            if DEBUG_Q:
                openoutfile()
                writeoutfile("Conv{}_input before preprocess(but after hardware_round):\n".format(g_conv_id))
                N= input.shape[0]
                C= input.shape[1]
                H=input.shape[2]
                W= input.shape[3]
                writeoutfile("input:Cout={},Cin={},K1={},K2={}\n".format(N, C, H,W))
                for n in range(N):
                    for h in range(H):
                        for w in range(W):
                            for c in range(C):
                                writeoutfile("{},".format(int(int_input[n][c][h][w])))
                            writeoutfile("\n")
                closeoutfile()
            
            ####int_input = clamp(int_input, c_round(-1 * self.act_max / self.act_scale), c_round(self.act_max / self.act_scale))  ###为适配硬件而注释掉
            # int_input = clamp(int_input, c_round(-1 * self.act_max / self.act_scale), c_round(self.act_max / self.act_scale))  ###为适配硬件而注释掉
            
            #print('self.act_max / self.act_scale={}'.format(c_round(self.act_max / self.act_scale)))
            if DEBUG_Q:
                openoutfile()
                writeoutfile("Conv{}_input   after clamp:\n".format(g_conv_id))
                N= input.shape[0]
                C= input.shape[1]
                H=input.shape[2]
                W= input.shape[3]
                writeoutfile("input:Cout={},Cin={},K1={},K2={}\n".format(N, C, H,W))
                for n in range(N):
                    for h in range(H):
                        for w in range(W):
                            for c in range(C):
                                writeoutfile("{},".format(int(int_input[n][c][h][w])))
                            writeoutfile("\n")
                closeoutfile()
            # convolution with interger input and param
            # 使用量化后的整数输入和整数权重计算卷积结果
            #print('g_conv_id={}'.format(g_conv_id))
            # print("int_input min:{} max:{}".format(int_input.min(), int_input.max()))
            int_output = F.conv2d(int_input, self.int_weight, self.int_bias, self.stride, self.padding,
                                                        self.dilation, self.groups)
            #############################
            is_overflow(int_output, self.bias_bits)
            ###########################
            
            if DEBUG_Q:
                N = int_output.shape[0]
                C = int_output.shape[1]
                H = int_output.shape[2]
                W = int_output.shape[3]
                openoutfile()
                writeoutfile("int_output after F.conv2d, Conv{},size: N={},C={},H={},W={}\n".format(g_conv_id,N, C, H, W))
                for n in range(N):
                    for h in range(H):
                        for w in range(W):
                            for c in range(C):
                                writeoutfile("{},".format(int(int_output[n][c][h][w])))
                            writeoutfile("\n")
                closeoutfile()
            
            #clamp_range = c_round(2 ** (c_round(self.bias_bits) - 1) - 1)
            #int_output = clamp(int_output,  min=-1*clamp_range, max=clamp_range) #为适配硬件而去掉
            #print('int_output0:{}'.format(int_output))

            # # 卷积结果后处理
            M_expand = self.M.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(int_output)
            pow_n_expand = c_round(torch.pow(2, c_round(self.n)))
            int_output = hardware_round(int_output * M_expand / pow_n_expand)

            clamp_range =self.out_clamp_range #c_round(2 ** (c_round(self.weight_bits) - 1) - 1)
            int_output = clamp(int_output,  min=-1*clamp_range, max=clamp_range)
            #print('int_output:{}'.format(int_output))
            # print("int_output min:{} max:{}".format(int_output.min(), int_output.max()))
            
            # 对输出层的卷积后处理结果进行处理，还原为浮点数
            if self.classify_layer:
                # convert output from interger to float 
                clamp_range =self.out_clamp_range # c_round(2 ** (c_round(self.weight_bits) - 1) - 1)
                pred_scale = self.pred_max / clamp_range
                if DEBUG_Q:
                    openoutfile()
                    writeoutfile("Conv{}:int:last pred_scale:{}\n".format(g_conv_id, pred_scale))
                    # print("Conv{}:last pred_scale:{}\n".format(g_conv_id, pred_scale))
                    N = int_output.shape[0]
                    C = int_output.shape[1]
                    H = int_output.shape[2]
                    W = int_output.shape[3]
                    writeoutfile("size: N={},C={},H={},W={}\n".format(N, C, H, W))
                    for n in range(N):
                        for h in range(H):
                            for w in range(W):
                                for c in range(C):
                                    writeoutfile("{},".format(int(int_output[n][c][h][w])))
                                writeoutfile("\n")
                    closeoutfile()
                 
                float_output = int_output * pred_scale
                # print("Conv{}".format(g_conv_id))
                # print("self.pred_max: ", self.pred_max)
                # print("clamp_range: ", clamp_range)
                # print("pred_scale: ", pred_scale)

                if DEBUG_Q:
                    openoutfile()
                    writeoutfile("Conv{}:float:last pred_scale:{}\n".format(g_conv_id, pred_scale))
                    # print("Conv{}:last pred_scale:{}\n".format(g_conv_id, pred_scale))
                    N = float_output.shape[0]
                    C = float_output.shape[1]
                    H = float_output.shape[2]
                    W = float_output.shape[3]
                    writeoutfile("size: N={},C={},H={},W={}\n".format(N, C, H, W))
                    for n in range(N):
                        for h in range(H):
                            for w in range(W):
                                for c in range(C):
                                    writeoutfile("{:.5f},".format(float(float_output[n][c][h][w])))
                                writeoutfile("\n")
                    closeoutfile()

                g_conv_id=(g_conv_id+1)%N_layers
                #print(float_output)
                return float_output
            if DEBUG_Q:
                openoutfile()

                N = int_output.shape[0]
                C = int_output.shape[1]
                H = int_output.shape[2]
                W = int_output.shape[3]
                writeoutfile("Conv{},N={},C={},H={},W={}\n".format((g_conv_id),N, C, H, W))
                # print("Conv{},size: N={},C={},H={},W={}".format((g_conv_id),N, C, H, W))
                # print('clamp_range={}'.format(clamp_range))

                for h in range(H):
                    for w in range(W):
                        for c in range(C):
                            writeoutfile("{},".format(int(int_output[0][c][h][w])))
                        writeoutfile("\n")
                closeoutfile()
             
            #######################
            '''
            if g_conv_id==17: #####转化为浮点，要用浮点与另一分支的浮点相加 
                #不珂使用bias_scale,不准
                next_act_scale = self.next_act_scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(int_output).detach()
                int_output=int_output*next_act_scale
            '''
            ###################  
            if DEBUG_Q:
                N = int_output.shape[0]
                C = int_output.shape[1]
                H = int_output.shape[2]
                W = int_output.shape[3]
                openoutfile()
                writeoutfile("int_output Conv{},size: N={},C={},H={},W={}\n".format(g_conv_id,N, C, H, W))
                for n in range(N):
                    for h in range(H):
                        for w in range(W):
                            for c in range(C):
                                writeoutfile("{},".format(int(int_output[n][c][h][w])))
                            writeoutfile("\n")
                closeoutfile()
            g_conv_id=(g_conv_id+1)%N_layers
            return int_output
        
class LeakyReLU_quantization(nn.LeakyReLU):
    """
    custom LeakyReLU layers for quantization
    """
    def __init__(self, negative_slope, inplace):
        super(LeakyReLU_quantization, self).__init__(negative_slope, inplace)
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input):
        output = F.leaky_relu(input, self.negative_slope, self.inplace)

        return Hardware_RoundFunction.apply(output)

g_act_scale_list = []
def update_next_act_scale(model):
    # 设置每层卷积层的下一层卷积层的激活值scale
    if len(g_act_scale_list)==0:
        for name, module in model.named_modules():
            if isinstance(module, Conv2d_quantization):
                g_act_scale_list.append(module.act_scale.detach().clone())
    else:
        g_act_scale_list.clear()
        for name, module in model.named_modules():
            if isinstance(module, Conv2d_quantization):
                g_act_scale_list.append(module.act_scale.detach().clone())
                
    replace_next_act_scale(model)
 
           
def replace_next_act_scale(model):
    # 设置每层卷积层的下一层卷积层的激活值scale
    from models.adaptive.resnet import AdaptiveBasicBlock
    '''
    act_scale_list = []
    for name, module in model.named_modules():
        if isinstance(module, Conv2d_quantization):
            act_scale_list.append(module.act_scale.detach().clone())
    '''
    '''
    if len(g_act_scale_list)==0:
        update_next_act_scale(model)
    '''    
    #next_scale_index = [1, 2, 3, 4, 5, 6, 7, 8, 12, 10, 11]
    #global N_layers
    
    index = 0
    block_act=[]
    for name, module in model.named_modules():
        if isinstance(module,AdaptiveBasicBlock):
            block_act.append(module.conv1.act_scale)
            index=index+1
    
    if hasattr(model,'module'):#####DDP
        block_act.append(model.module.fc.act_scale)#########################
        model.module.conv1.next_act_scale=block_act[0].detach().clone()
    else:
        block_act.append(model.fc.act_scale)#########################
        model.conv1.next_act_scale=block_act[0].detach().clone()
        
    index=0
    for name, module in model.named_modules(): 
        if isinstance(module,AdaptiveBasicBlock):
            module.conv1.next_act_scale=module.conv2.act_scale.detach().clone()
            module.conv2.next_act_scale=block_act[index+1].detach().clone()
            if module.downsample is not None: 
                module.downsample[0].next_act_scale=block_act[index+1].detach().clone()
            
            index=index+1    
     
            


def replace_layer_by_unique_name(module, unique_name, layer):
    # 替换一层原始的卷积层为自定义层 如模拟量化的卷积层
    unique_names = unique_name.split(".")
    if len(unique_names) == 1:
        #print("unique_names[0]:",unique_names[0])
        #print("layer:",layer)
        #print("module._modules:",module._modules)        
        module._modules[unique_names[0]] = layer
    else:
        #print("unique_names:",unique_names)
        #print("layer:",layer)
        replace_layer_by_unique_name(
            module._modules[unique_names[0]], #这里把第几层哪个函数进行递归分解
            ".".join(unique_names[1:]),
            layer)


# replace model
def replace(model, quantization_bits=8, m_bits=12, bias_bits=16, inference_type="all_fp", Mn_aware=False, fuseBN=False):
    # 遍历整个模型，把原始模型的卷积层替换为自定义的模拟量化卷积层
    global N_layers
    count = 0
    set_bias_bits = bias_bits
    last_layer=[N_layers-1]
    print(last_layer)
    for name, module in model.named_modules(): 
        # first layer and last layer quantized to 8 bit if quantization_bits < 8 
        if quantization_bits == 4 and count in [0,last_layer[0]]:
            if count==0:
                act_bits =8 #6 # 4
                weight_bits =8 #6 
            elif count==last_layer[0]:
                act_bits =4 # 4
                weight_bits =4 
            bias_bits =16 
        else:
            act_bits = quantization_bits
            weight_bits = quantization_bits
            bias_bits = set_bias_bits

        # print("set layer: {} act_bits: {}  weight_bits:{}  bias_bits:{}".format(count, act_bits, weight_bits, bias_bits))

        if  isinstance(module, nn.Conv2d):
            # print('Conv{}.............'.format(count))
            # print('in{},out{}'.format(module.in_channels,module.out_channels))
            temp_conv = Conv2d_quantization(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                groups=module.groups,
                bias=(module.bias is not None),
                act_bits=act_bits, weight_bits=weight_bits, m_bits=m_bits, bias_bits=bias_bits, 
                inference_type=inference_type,
                classify_layer=True if count in last_layer else False,
                fuseBN=False if count in last_layer else fuseBN,#######the last layer doesn't have BN layer
                Mn_aware=Mn_aware
            )
            temp_conv.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                temp_conv.bias.data.copy_(module.bias.data)
            replace_layer_by_unique_name(model, name, temp_conv)
            count += 1
    # print("After replace:\n {}".format(model))
    return model

def replace_alpha(model, bit, check_multiplier=False):
    # 手动设置alphs，因为太繁琐且不够自动化，已弃用
    # changle alpha
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, Conv2d_quantization):

            if bit == 4:
                # if count == 0:
                #     module.alpha = module.alpha - module.alpha + 3.0
                #     print("replace module.alpha: {}".format(module.alpha))
                # if count == 12:
                #     module.alpha = module.alpha - module.alpha + 14.0
                #     print("replace module.alpha: {}".format(module.alpha))
                pass

            if bit == 6:
                # if count == 0:
                #     module.alpha = module.alpha - module.alpha + 3.3
                #     print("replace module.alpha: {}".format(module.alpha))
                # if count == 1:
                #     module.alpha = module.alpha - module.alpha + 1.1
                #     print("replace module.alpha: {}".format(module.alpha))
                # if count == 2:
                #     module.alpha = module.alpha - module.alpha + 7.0
                #     print("replace module.alpha: {}".format(module.alpha))
                # if count == 3:
                #     module.alpha = module.alpha - module.alpha + 1.4
                #     print("replace module.alpha: {}".format(module.alpha))
                # if count == 4:
                #     module.alpha = module.alpha - module.alpha + 1.1
                #     print("replace module.alpha: {}".format(module.alpha))
                # if count == 5:
                #     module.alpha = module.alpha - module.alpha + 1.1
                #     print("replace module.alpha: {}".format(module.alpha))
                # if count == 6:
                #     module.alpha = module.alpha - module.alpha + 1.5
                #     print("replace module.alpha: {}".format(module.alpha))
                # if count == 6:
                #     module.alpha = module.alpha - module.alpha + 1.2
                #     print("replace module.alpha: {}".format(module.alpha))
                # if count == 10:
                #     module.alpha = module.alpha - module.alpha + 1.4
                #     print("replace module.alpha: {}".format(module.alpha))
                # if count == 11:
                #     module.alpha = module.alpha - module.alpha + 2.2
                #     print("replace module.alpha: {}".format(module.alpha))
                # if count == 12:
                #     module.alpha = module.alpha - module.alpha + 2.2
                #     print("replace module.alpha: {}".format(module.alpha))
                pass

            if bit == 14:
                pass

            count += 1


def replace_LeakyReLU(model):
    # 遍历整个模型，把原始模型的LeakyReLU层替换为自定义的量化LeakyReLU层
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.LeakyReLU):
            temp_LeakyReLU = LeakyReLU_quantization(negative_slope=module.negative_slope, inplace=module.inplace)
            replace_layer_by_unique_name(model, name, temp_LeakyReLU)
    
    #print("After replace:\n {}".format(model))
    return model

def calculate_sign_overflow_number(x, bit):
    min = - 2 ** (bit - 1) + 1
    max = 2 ** (bit - 1) - 1
    # print(min, max)
    # print('calculate_sign_overflow_number', torch.min(x), torch.max(x))
    down_overflow_index = x < min
    up_overflow_index = x > max
    No = torch.sum(down_overflow_index) + torch.sum(up_overflow_index)
    return No

def layer_transform(inference_type, model):
    input_scale=None
    conv_id=0
    # set M and n to conv
    if inference_type=='full_int':
        for name, layer in model.named_modules():
            # print(name)
             
            if isinstance(layer, Conv2d_quantization):
                # print(layer.fuseBN)
                # print(layer.weight.shape)
                if layer.bias is not None:
                    print(layer.bias.shape)

                if layer.in_channels==3:
                    input_scale=layer.act_scale.clone()
                
                #fuse_conv_and_bn()函数中
                
                # model param
                if layer.fuseBN:#layer.classify_layer:
                    print('True.............')
                    #bias=layer.bias
                    #weight=layer.weight
                #else:
                    ######################## fuseBN ##############################
                    if hasattr(layer,'beta'):
                        if layer.bias is not None:
                            bias = reshape_to_bias(layer.beta + (layer.bias - layer.running_mean) * (
                                layer.gamma / torch.sqrt(layer.running_var + layer.eps)))
                        else:
                            bias = reshape_to_bias(layer.beta - layer.running_mean * (
                                layer.gamma / torch.sqrt(layer.running_var + layer.eps)))  # b铻峳unning
                            ######################
                            # layer.bias=nn.Parameter(torch.Tensor(layer.out_channels))
                            ###################
                            
                        weight = layer.weight * reshape_to_weight(
                                layer.gamma / torch.sqrt(layer.running_var + layer.eps))  # w铻峳unning
                    else:
                        bias = layer.bias
                        weight = layer.weight
                else:
                    ###layer.fuseBN=False
                    bias = layer.bias
                    weight = layer.weight

                # quantized param
                #######################################
                #######################################   
                print('conv_id={}'.format(conv_id))
                
                # layer.bias.data.copy_(bias)
                # layer.weight.data.copy_(weight)#####    

                # 20240204改
                layer.fused_bias = bias
                layer.fused_weight = weight #####   
                  
               
                if b_split_2part and conv_id < int_start_conv:
                    print('conv_id{}'.format(conv_id))
                    conv_id+=1
                    continue
                #######################################   
                #######################################
                
                weight_bits = layer.weight_bits
                bias_bits = layer.bias_bits

                # scale
                weight_scale = layer.weight_scale

                # process weight: per-channel
                weight_scale_expand = weight_scale.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(weight)
                quantized_weight = c_round(weight / weight_scale_expand)
                quantized_weight = torch.clamp(quantized_weight, min=int(c_round(-2 ** (c_round(weight_bits) - 1) + 1)),
                                               max=int(c_round(2 ** (c_round(weight_bits) - 1) - 1)))

                # Test weight
                No = calculate_sign_overflow_number(quantized_weight, weight_bits)
                if No != 0:
                    print('No={}, overflow !!!'.format(No))
                    assert False, "quantzied weight error !!!"

                # process bias
                if bias is not None:
                    # process bias
                    bias_scale = layer.bias_scale
                    quantized_bias = c_round(bias / bias_scale)
                    # Test bias
                    No = calculate_sign_overflow_number(quantized_bias, bias_bits)
                    
                    if No != 0:
                        print(No)
                        assert False, "quantized bias error !!!"

                # replace weight
                layer.int_weight = nn.Parameter(quantized_weight)

                # replace bias
                layer.int_bias = nn.Parameter(quantized_bias)
                conv_id+=1

                # print("layer.M.max():{}".format(layer.M.max()))
                # print("layer.n.max():{}".format(layer.n.max()))

    return input_scale
if __name__ == "__main__":

    pass
