import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
import torch.distributed as dist
from unet.unet_parts import *

def c_round(n):
    # 四舍五入
    return torch.where(n>0.0, torch.floor(n+0.5), torch.ceil(n-0.5))

def hardware_round(n):
    # 硬件上的四舍五入
    return torch.floor(n+0.5)

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

        minv = - maxv
        scale = (maxv - minv) / (num_levels - 2)

        if alpha is not None:
            scale = scale * alpha

        # clamp
        x = clamp(x, min=minv, max=maxv)
        # quantize
        x_int = Hardware_RoundFunction.apply(x / scale)

        # clamp
        num_levels = num_levels
        x_int = torch.clamp(x_int, -1 * (num_levels / 2 - 1), (num_levels / 2 - 1))

        # dequantize
        x_dequant = x_int * scale

        return x_dequant, scale

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
        # num_levels = int(num_levels)
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
        scale = (maxv - minv) / (num_levels - 2)   # scale shape = c_out*1

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
        x = clamp(x, min=minv, max=maxv)

        # quantize
        x_int = RoundFunction.apply(x / scale_expand)

        # clamp
        num_levels = num_levels
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


def cal_M_n(act_scale, weight_scale, next_act_scale, m_bits, max_M):

    # 计算卷积后处理的M和n
    M_fp = act_scale * weight_scale / next_act_scale
    n = torch.floor(torch.log2(max_M / M_fp))

    # 取最小的n
    n = torch.tensor([c_round(torch.min(n))],device=M_fp.device)

    # n必须小于或等于31，因为硬件中使用5比特存储n
    if n > 31:
        n = n - n + 31

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


def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)


def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)

def reshape_to_bias(input):
    return input.reshape(-1)

def fuse_conv_and_bn(conv, bn):

    with torch.no_grad():

        conv.eps = 1e-5
        conv.momentum = 0.01
        conv.gamma = nn.Parameter(torch.Tensor(conv.out_channels))
        conv.beta = nn.Parameter(torch.Tensor(conv.out_channels))
        conv.register_buffer('running_mean', torch.zeros(conv.out_channels))
        conv.register_buffer('running_var', torch.ones(conv.out_channels))

        conv.running_mean.copy_(bn.running_mean)
        conv.running_var.copy_(bn.running_var)
        conv.gamma.copy_(bn.weight)
        conv.beta.copy_(bn.bias)
        conv.fuseBN=True

    return conv

def fuse_doubleconv(net):
    for name, module in net.named_modules():
        if isinstance(module, DoubleConv):
            module.double_conv[0] = fuse_conv_and_bn(module.double_conv[0], module.double_conv[1])
            module.double_conv[3] = fuse_conv_and_bn(module.double_conv[3], module.double_conv[4])
            del module.double_conv[4]
            del module.double_conv[1]
    return net

def open_Mn(net, m_bits):
    for name, module in net.named_modules():
        if isinstance(module, Conv2d_quantization):
        # if(name=='model.24.m.2'):
            module.Mn_aware=True
            module.register_buffer('M', torch.zeros(module.out_channels))
            module.register_buffer('n', torch.zeros(1))
            module.m_bits = module.m_bits - module.m_bits + m_bits
            module.max_M = module.max_M - module.max_M + c_round(2 ** module.m_bits - 1)
            if module.classify_layer:
                module.pred_max = nn.Parameter(torch.ones(1)*20)
                # module.pred_max = nn.Parameter(torch.ones(1) * 12)
    return net

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

class Conv2d_quantization(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 act_bits=16, weight_bits=16, m_bits=12, bias_bits=32, inference_type="all_fp",
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
        self.register_buffer('act_scale', torch.ones(1)/127.5)

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
        self.act_max = nn.Parameter(torch.ones(1)*10)

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
                # self.pred_max = nn.Parameter(torch.ones(1)*20)
                self.pred_max = nn.Parameter(torch.ones(1)*20)


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

        # 训练的时候使用这个模式，用于模拟量化的过程
        if self.inference_type == "all_fp":

            # float inference
            if self.fuseBN:
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
                    # print("testing")
                    # BN融合
                    if self.bias is not None:
                        bias = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (
                            self.gamma / torch.sqrt(self.running_var + self.eps)))
                    else:
                        bias = reshape_to_bias(self.beta - self.running_mean * (
                            self.gamma / torch.sqrt(self.running_var + self.eps)))  # b融running
                    weight = self.weight * reshape_to_weight(
                        self.gamma / torch.sqrt(self.running_var + self.eps))  # w融running

                # 对卷积层的输入进行模拟量化
                quantized_input, self.act_scale = uniform_symmetric_quantizer_per_tensor\
                    (input, num_levels=self.act_num_levels, bits=self.act_bits, cal_max=self.act_max, alpha=self.alpha)

                # 对卷积层的权重进行模拟量化
                quantized_weight, self.weight_scale = uniform_symmetric_quantizer_per_channel\
                    (weight, num_levels=self.weight_num_levels, bits=self.weight_bits, alpha=self.alpha, bias=bias,
                        acc_clamp_range=self.acc_clamp_range, act_scale=self.act_scale)

                # 对卷积层的bias进行模拟量化
                # quantization self.bias
                if self.bias is not None:
                    self.bias_scale = self.act_scale * self.weight_scale
                    quantized_bias = uniform_symmetric_quantizer_per_channel\
                        (bias, num_levels=self.bias_num_levels, bits=self.bias_bits, bias_scale=self.bias_scale)
                else:
                    # not quantization self.bias
                    quantized_bias = None

                # 使用模拟量化后的输入和参数计算卷积结果
                if self.training:
                    output = F.conv2d(quantized_input, quantized_weight, None, self.stride, self.padding,
                                      self.dilation, self.groups)
                    output = output * reshape_to_activation(
                        torch.sqrt(self.running_var + self.eps) / torch.sqrt(batch_var + self.eps))
                    output = output + reshape_to_activation(quantized_bias)

                else:
                    output = F.conv2d(quantized_input, quantized_weight, quantized_bias, self.stride, self.padding,
                                      self.dilation, self.groups)

            else:
                # 对卷积层的输入进行模拟量化
                quantized_input, self.act_scale = uniform_symmetric_quantizer_per_tensor\
                    (input, num_levels=self.act_num_levels, bits=self.act_bits, cal_max=self.act_max, alpha=self.alpha)

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


            # 对输出层进行卷积后处理仿真训练
            if self.classify_layer and self.Mn_aware:
                # convert fp32 to int16
                bias_scale_expand = self.bias_scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(output).detach()
                output = RoundFunction.apply(output / bias_scale_expand)
                self.tmp_int_output2 = output

                output = clamp(output, min=-1 * self.acc_clamp_range, max=self.acc_clamp_range)

                # convert int16 to int8
                pred_scale = self.pred_max / self.out_clamp_range

                self.M, self.n, n_pow = cal_M_n(self.act_scale, self.weight_scale, pred_scale, self.m_bits, self.max_M)

                M_expand = self.M.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(output).detach()
                output = Hardware_RoundFunction.apply(output * M_expand / n_pow)

                output = clamp(output, min=-1 * self.out_clamp_range, max=self.out_clamp_range)

                # convert int8 to fp32
                output = output * pred_scale

            # 对融合Bn后的卷积层进行卷积后处理仿真训练
            elif self.fuseBN and self.Mn_aware:
                # convert fp32 to int16
                bias_scale_expand = self.bias_scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(output).detach()
                output = RoundFunction.apply(output / bias_scale_expand)

                output = clamp(output,  min=-1*self.acc_clamp_range, max=self.acc_clamp_range)

                # convert int16 to int8
                self.M, self.n, n_pow = cal_M_n(self.act_scale, self.weight_scale, self.next_act_scale, self.m_bits, self.max_M)

                M_expand = self.M.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(output).detach()
                # output = hardware_round(output * M_fp_expand)

                output = Hardware_RoundFunction.apply(output * M_expand / n_pow)

                output = clamp(output,  min=-1*self.out_clamp_range, max=self.out_clamp_range)

                # convert int8 to fp32
                output = output * self.next_act_scale

            return output

        if self.inference_type == "full_int":
            print("test_min", input.min())
            print("test_max", input.max())
            
            if hasattr(self, 'convert_fp32_to_int'):
                int_input = hardware_round(input/self.act_scale)
            else:
                int_input = hardware_round(input)

            int_input = clamp(int_input, c_round(-1 * self.act_max / self.act_scale),
                              c_round(self.act_max / self.act_scale))

            # convolution with interger input and param
            # 使用量化后的整数输入和整数权重计算卷积结果
            int_output = F.conv2d(int_input, self.int_weight, self.int_bias, self.stride, self.padding,
                                                        self.dilation, self.groups)

            clamp_range = c_round(2 ** (c_round(self.bias_bits) - 1) - 1)
            # int_output = clamp(int_output, min=-1 * self.acc_clamp_range, max=self.acc_clamp_range)
            self.tmp_int_output=int_output
            int_output = clamp(int_output,  min=-1*clamp_range, max=clamp_range)

            # temp code
            # n必须小于或等于31，因为硬件中使用5比特存储n
            if self.n > 31:
                self.M, self.n, n_pow = cal_M_n(self.act_scale, self.weight_scale, self.next_act_scale, self.m_bits, self.max_M)
            # temp code

            # # 卷积结果后处理
            M_expand = self.M.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(int_output)
            pow_n_expand = c_round(torch.pow(2, c_round(self.n)))
            int_output = hardware_round(int_output * M_expand / pow_n_expand)

            clamp_range = c_round(2 ** (c_round(self.weight_bits) - 1) - 1)
            int_output = clamp(int_output,  min=-1*clamp_range, max=clamp_range)

            # 对输出层的卷积后处理结果进行处理，还原为浮点数
            if self.classify_layer:
                # convert output from interger to float
                clamp_range = c_round(2 ** (c_round(self.weight_bits) - 1) - 1)
                pred_scale = self.pred_max / clamp_range

                float_output = int_output * pred_scale

                return float_output

            return int_output

def replace_next_act_scale(model):
    # 设置每层卷积层的下一层卷积层的激活值scale
    act_scale_list = []
    for name, module in model.named_modules():
        if isinstance(module, Conv2d_quantization):
            act_scale_list.append(module.act_scale.detach().clone())

    index=0
    for name, module in model.named_modules():
        if isinstance(module, Conv2d_quantization):
            module.next_act_scale = act_scale_list[index+1].detach().clone()

            index += 1
            if index >= len(act_scale_list)-1:
                break

def replace_layer_by_unique_name(module, unique_name, layer):
    # 替换一层原始的卷积层为自定义层 如模拟量化的卷积层
    unique_names = unique_name.split(".")
    if len(unique_names) == 1:
        module._modules[unique_names[0]] = layer
    else:
        replace_layer_by_unique_name(
            module._modules[unique_names[0]],
            ".".join(unique_names[1:]),
            layer)


# replace model
def replace(model, quantization_bits=4, m_bits=12, bias_bits=32, inference_type="all_fp", Mn_aware=True, fuseBN=False, layer_bit_dict=None):
    # 遍历整个模型，把原始模型的卷积层替换为自定义的模拟量化卷积层
    if layer_bit_dict is None:
        count = 0
        for name, module in model.named_modules():
            bits_act = 4 if count == 0 and quantization_bits<6 else quantization_bits
            bits_weight = 4 if count in [0,18] and quantization_bits<6 else quantization_bits
            bits_bias = 16 if count in [0,18] and bias_bits<16 else bias_bits

            # print(name)
            if(name=='model.24.m.2' or name=='model.model.24.m.2'):
                if isinstance(module, nn.Conv2d):
                    # print(module)
                    # break
                    temp_conv = Conv2d_quantization(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    groups=module.groups,
                    bias=(module.bias is not None),
                    act_bits=bits_act,
                    weight_bits=bits_weight,
                    m_bits=m_bits,
                    bias_bits=bits_bias,
                    inference_type=inference_type,
                    # classify_layer=True if module.out_channels==20 else False,
                    classify_layer=True,
                    fuseBN=False if module.out_channels==20 else fuseBN,
                    Mn_aware=Mn_aware)
                    
                    temp_conv.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        temp_conv.bias.data.copy_(module.bias.data)

                    # reset the first conv layer's out_clamp_range as 4bit
                    if count == 0 and quantization_bits < 6:
                        temp_conv.out_clamp_range = temp_conv.out_clamp_range - temp_conv.out_clamp_range + c_round(
                            2 ** (c_round(torch.tensor([4])) - 1) - 1)

                    replace_layer_by_unique_name(model, name, temp_conv)
                    count += 1

                    print("set {}th conv layer: {}  act_bits: {}  weight_bits:{}  bias_bits:{}".format
                        (count, name, bits_act, bits_weight,bits_bias))

            

    # 16bit and 8bit mixture quantization
    elif layer_bit_dict is not None:
        count = 0
        previous_layer_output_bit = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if name in layer_bit_dict.keys():

                    # 确保输入特征的bit数于上一层的输出特征的bit数一致，比如上一层输出特征是8bit，那么当前层的输入特征也是按照8bit处理
                    if layer_bit_dict[name] == 16:
                        if previous_layer_output_bit is not None:
                            bits_act = previous_layer_output_bit
                        else:
                            bits_act = 16
                        bits_weight = 16
                        bits_bias = 32


                    elif layer_bit_dict[name] == 8:
                        bits_act = 8
                        bits_weight = 8
                        bits_bias = 16
                    else:
                        print("Unsupport bit:{}".format(layer_bit_dict[name]))
                        assert False

                    previous_layer_output_bit = bits_weight

                else:
                    print("name:{} not in layer_bit_dict.keys(), unknow quantization bit!!!".format(name))
                    print("layer_bit_dict.keys():{}".format(layer_bit_dict.keys()))
                    assert False

                temp_conv = Conv2d_quantization(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    groups=module.groups,
                    bias=(module.bias is not None),
                    act_bits=bits_act,
                    weight_bits=bits_weight,
                    m_bits=m_bits,
                    bias_bits=bits_bias,
                    inference_type=inference_type,
                    classify_layer=True if module.out_channels == 20 else False,
                    fuseBN=False if module.out_channels == 20 else fuseBN,
                    Mn_aware=Mn_aware
                )
                temp_conv.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    temp_conv.bias.data.copy_(module.bias.data)

                # # reset the first conv layer's out_clamp_range as 4bit
                # if count == 0 and quantization_bits < 6:
                #     temp_conv.out_clamp_range = temp_conv.out_clamp_range - temp_conv.out_clamp_range + c_round(
                #         2 ** (c_round(torch.tensor([4])) - 1) - 1)

                replace_layer_by_unique_name(model, name, temp_conv)
                count += 1

                print("set {}th conv layer: {}  act_bits: {}  weight_bits:{}  bias_bits:{}".format
                      (count, name, bits_act, bits_weight, bits_bias))

        reset_out_clamp_range(model)

        for name, module in model.named_modules():
            if isinstance(module, Conv2d_quantization):
                print("name: {}  weight_bits:{} out_clamp_range:{}".format(name, module.weight_bits, module.out_clamp_range))

    return model


def reset_out_clamp_range(model):
    # 设置每层卷积层的下一层卷积层的激活值scale
    out_clamp_range_list = []
    for name, module in model.named_modules():
        if isinstance(module, Conv2d_quantization):
            out_clamp_range_list.append(c_round(2 ** (c_round(module.weight_bits) - 1) - 1).detach().clone())

    index=0
    for name, module in model.named_modules():
        if isinstance(module, Conv2d_quantization):
            if module.out_clamp_range > out_clamp_range_list[index+1]:
                module.out_clamp_range = module.out_clamp_range - module.out_clamp_range + out_clamp_range_list[index+1].detach().clone()

            index += 1
            if index >= len(out_clamp_range_list)-1:
                break


def calculate_sign_overflow_number(x, bit):
    min = - 2 ** (bit - 1) + 1
    max = 2 ** (bit - 1) - 1
    # print(min, max)
    # print('calculate_sign_overflow_number', torch.min(x), torch.max(x))
    down_overflow_index = x < min
    up_overflow_index = x > max
    No = torch.sum(down_overflow_index) + torch.sum(up_overflow_index)
    return No

def layer_transform(model):
    input_scale=None
    # set M and n to conv
    # if opt.inference_type=='full_int':
    for name, layer in model.named_modules():
        if isinstance(layer, Conv2d_quantization):
        # if(name=='model.24.m.2'):

            if layer.in_channels==3:
                input_scale=layer.act_scale.clone()

            # model param
            if layer.classify_layer:
                bias=layer.bias
                weight=layer.weight
            else:
                if layer.bias is not None:
                    bias = reshape_to_bias(layer.beta + (layer.bias - layer.running_mean) * (
                        layer.gamma / torch.sqrt(layer.running_var + layer.eps)))
                else:
                    bias = reshape_to_bias(layer.beta - layer.running_mean * (
                        layer.gamma / torch.sqrt(layer.running_var + layer.eps)))  # b融running
                weight = layer.weight * reshape_to_weight(
                    layer.gamma / torch.sqrt(layer.running_var + layer.eps))  # w融running

            # quantized param
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
                    assert False, "quantized bias error !!!"

            # replace weight
            layer.int_weight = nn.Parameter(quantized_weight)

            # replace bias
            layer.int_bias = nn.Parameter(quantized_bias)

            # print("layer.M.max():{}".format(layer.M.max()))
            # print("layer.n.max():{}".format(layer.n.max()))

    return model, input_scale


def calculate_weight_perturbation(model=None, param_name=None, bit=None):
    for name, param in model.named_parameters():
        if name == param_name:
            quantized_param, weight_scale = uniform_symmetric_quantizer_per_channel(param, num_levels=c_round(torch.Tensor([2 ** bit])), bits=bit)
            return ((param-quantized_param)*(param-quantized_param)).sum().cpu().item()  # the square of l2 norm of the difference ==> ||W-Q(W)||^2_2

    print("can not find the params named:{}".fromat(param_name))
    assert False

if __name__ == "__main__":

    pass
