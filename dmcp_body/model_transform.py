import argparse
import logging

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

import torch.backends.cudnn as cudnn 
import tflite_quantization_PACT_weight_and_act as tflite
import utils.tools as tools
from models.adaptive.resnet import AdaptiveBasicBlock 

def c_round(n):
    return torch.where(n>0.0, torch.floor(n+0.5), torch.ceil(n-0.5))

def hook_conv_results_checkoverflow(module, input, output):

    device = torch.device('cpu')

    # conv_accumulator_bits: [min, max], sign=True
    bias_bits_global = tflite.c_round(module.bias_bits)
    min = - 2 ** (bias_bits_global - 1) + 1
    max = 2 ** (bias_bits_global - 1) - 1

    scale = (module.act_scale * module.weight_scale).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(
        output)
    output = tflite.c_round(output.detach() / scale)
    down_overflow_index = output < min
    up_overflow_index = output > max
    No = (torch.sum(down_overflow_index) + torch.sum(up_overflow_index)).to(device)

    if No>0:
        print("############################ overflow happen ###################")
        print("overflow No: {}".format(No))
        print("overflow module: {}".format(module))
        print("module.alpha: {}".format(module.alpha))
        print("module.act_scale: {}".format(module.act_scale))
        print("output.min: {}".format(output.min()))
        print("output.max: {}".format(output.max()))
        print("max: {}".format(max))
        print("module.quant_bias.max: {}".format((module.bias / module.bias_scale).clone().max()))
        # print('module.bias: {}'.format(module.bias))
        # print('module.quant_bias: {}'.format((module.bias / module.bias_scale).clone()))



def calculate_sign_overflow_number(x, bit):
    min = c_round(- 2 ** (bit - 1) + 1)
    max = c_round(2 ** (bit - 1) - 1)
    # print(min, max)
    # print('calculate_sign_overflow_number', torch.min(x), torch.max(x))
    down_overflow_index = x < min
    up_overflow_index = x > max
    No = torch.sum(down_overflow_index) + torch.sum(up_overflow_index)
    return No

g_conv_id=0
layer_start_quantize=0#19
def layer_transform(name, layer, activation_scale_str, weight_scale_str, params):
    global g_conv_id
    global layer_start_quantize
    # model param
    # weight = layer.weight
    # bias = layer.bias

    # 20240204改
    weight = layer.fused_weight
    bias = layer.fused_bias

    # print('bias={}'.format(bias))

    # quantized param
    act_bits = c_round(layer.act_bits)
    weight_bits = c_round(layer.weight_bits)
    m_bits = c_round(layer.m_bits)
    bias_bits = c_round(layer.bias_bits)

    # scale
    act_scale = layer.act_scale
    weight_scale = layer.weight_scale
    print('act_bits={:d}, weight_bits={:d}, m_bits={:d}, bias_bits={:d}'
          ''.format(int(c_round(act_bits)),int(c_round(weight_bits)),int(c_round(m_bits)), int(c_round(bias_bits))))


    # process activation: per-tensor
    print('process activation: per-tensor')

    clamp_max = c_round(layer.act_max.cpu() / act_scale)

    # 新增的代码，因为用了power of two scale后，clamp max可能会超出低比特数的表示范围，所以需要判定一下，确定最终的clamp max
    bit_clamp_max = layer.act_num_levels / 2 - 1
    if clamp_max > bit_clamp_max:
        clamp_max = bit_clamp_max

    output_str = '{} {:.8f} {:d}\n'.format(name, float(act_scale), int(c_round(clamp_max)))
    activation_scale_str += output_str


    # process weight: per-channel
    print('process weight: per-channel')
    #quantized_weight = c_round(weight)
    # process weight: per-channel
    print('process weight: per-channel')
    weight_scale_expand = weight_scale.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(weight)
    quantized_weight = c_round(weight / weight_scale_expand)
    quantized_weight = torch.clamp(quantized_weight, min=int(c_round(-2 ** (weight_bits - 1) + 1)),
                                   max=int(c_round(2 ** (weight_bits - 1) - 1)))

    # Test weight
    # print(quantized_weight)
    No = calculate_sign_overflow_number(quantized_weight, weight_bits)
    if No != 0:
        print('No={}, overflow !!!'.format(No))
        assert False, "quantzied weight error !!!"

    for i in range(len(weight_scale)):
        output_str = '{} {} {:.8f}\n'.format(name, i, float(weight_scale[i]))
        weight_scale_str += output_str
    # assert False
    weight_name = '{} weight'.format(name)
    #params[weight_name] = quantized_weight.numpy().astype(int)
    ###################split to 2 parts: one part is not quantized,the other part is quantized ############################
    if g_conv_id < layer_start_quantize:
        params[weight_name] = layer.weight.detach().numpy().astype(float)
    else:
        params[weight_name] = quantized_weight.detach().numpy().astype(int)
    ###############################################
    # process bias
    print('process bias')
    # if bias is not None:
    if layer.int_bias is not None:    
        #quantized_bias = c_round(bias)
        quantized_bias = c_round(bias / layer.bias_scale)
        # Test bias
        No = calculate_sign_overflow_number(quantized_bias, bias_bits)
        print(quantized_bias.abs().max())
        print(2 ** (bias_bits - 1) - 1)
        if No != 0:
            print(No)

        bias_name = '{} bias'.format(name)
        #params[bias_name] = quantized_bias.numpy().astype(int)
        ##################split to 2 parts: one part is not quantized,the other part is quantized############################
        if g_conv_id < layer_start_quantize:
            params[bias_name] = layer.bias.detach().numpy().astype(float)
        else:
            params[bias_name] = quantized_bias.detach().numpy().astype(int)
        ##################################################
    else:
        quantized_bias =None
    g_conv_id += 1
    return activation_scale_str, weight_scale_str, params, act_scale, weight_scale, m_bits, name


def layer_transform_linear(name, layer, activation_scale_str, weight_scale_str, params):
    # model param
    weight = layer.int_weight
    bias = layer.int_bias
    # print('bias={}'.format(bias))

    # quantized param
    act_bits = c_round(layer.act_bits)
    weight_bits = c_round(layer.weight_bits)
    m_bits = c_round(layer.m_bits)
    bias_bits = c_round(layer.bias_bits)

    # scale
    act_scale = layer.act_scale
    weight_scale = layer.weight_scale
    print('act_bits={:d}, weight_bits={:d}, m_bits={:d}, bias_bits={:d}'
          ''.format(int(c_round(act_bits)),int(c_round(weight_bits)),int(c_round(m_bits)), int(c_round(bias_bits))))


    # process activation: per-tensor
    print('process activation: per-tensor')

    clamp_max = c_round(layer.act_max.cpu() / act_scale)

    # 新增的代码，因为用了power of two scale后，clamp max可能会超出低比特数的表示范围，所以需要判定一下，确定最终的clamp max
    bit_clamp_max = layer.act_num_levels / 2 - 1
    if clamp_max > bit_clamp_max:
        clamp_max = bit_clamp_max
        
    output_str = '{} {:.8f} {:d}\n'.format(name, float(act_scale), int(c_round(clamp_max)))
    activation_scale_str += output_str 

    # process weight: per-channel
    print('process weight: per-channel')
    quantized_weight = c_round(weight)

    # Test weight
    # print(quantized_weight)
    No = calculate_sign_overflow_number(quantized_weight, weight_bits)
    if No != 0:
        print('No={}, overflow !!!'.format(No))
        assert False, "quantzied weight error !!!"

    for i in range(len(weight_scale)):
        output_str = '{} {} {:.8f}\n'.format(name, i, float(weight_scale[i]))
        weight_scale_str += output_str
    # assert False
    weight_name = '{} weight'.format(name)
    params[weight_name] = quantized_weight.detach().numpy().astype(int)

    # process bias
    print('process bias')
    quantized_bias = c_round(bias)
    # Test bias
    No = calculate_sign_overflow_number(quantized_bias, bias_bits)
    if No != 0:
        assert False, "quantized bias error !!!"

    bias_name = '{} bias'.format(name)
    params[bias_name] = quantized_bias.detach().numpy().astype(int)

    return activation_scale_str, weight_scale_str, params, act_scale, weight_scale, m_bits, name


def get_scale_approximation_shift_bits(fp32_scale, mult_bits):

    shift_bits = torch.floor(torch.log2((2 ** mult_bits - 1) / fp32_scale))
    # print('shift_bits.shape={},\nshift_bits={}'.format(shift_bits.shape, shift_bits))
    shift_bits = torch.min(mult_bits, shift_bits)
    # print('shift_bits.shape={},\nshift_bits={}'.format(shift_bits.shape, shift_bits))
    return shift_bits

def get_scale_approximation_mult(fp32_scale, shift_bits):
    return torch.floor(fp32_scale * (2 ** shift_bits))


def cal_multiplier(model, act_scale_list, weight_scale_list, m_bits_list, conv_name_list, multiplier_scale_str,
                   layer_list):
    count = 0
    power_of_two_scale=False

    for name, layer in model.named_modules():
        # if isinstance(layer, tflite.Conv2d_quantization) or isinstance(layer, tflite.Linear_quantization):
        if isinstance(layer, tflite.Conv2d_quantization): 
            #if not layer.power_of_two_scale:
            if not power_of_two_scale:
                M_shift = c_round(layer.n)
                M_multiplier_int = c_round(layer.M)
                M_approximation_float = M_multiplier_int / c_round(2 ** M_shift)

                if M_approximation_float.min() == 0:
                    print("The Channel with M=0 in {} should be removed!!!".format(name))
                    # assert False

                max = 2 ** (m_bits_list[count]) - 1
                up_overflow_index = M_multiplier_int > max
                No = torch.sum(up_overflow_index)
                if No > 0:
                    print("############################ M overflow happen ###################")
                    print("M overflow No: {}".format(No))
                    assert False

                for i in range(len(M_approximation_float)):
                    output_str = '{} {} {:.8f} {:d} {:d}\n'.format(conv_name_list[count], i,
                                                                   float(M_approximation_float[i]),
                                                                   int(c_round(M_shift)), int(c_round(M_multiplier_int[i])))
                    multiplier_scale_str += output_str

                count += 1

            else:
                M_shift = c_round(layer.n)

                for i in range(len(M_shift)):
                    output_str = '{} {} {:d}\n'.format(conv_name_list[count], i, int(c_round(M_shift[i])))
                    multiplier_scale_str += output_str

                count += 1

    return multiplier_scale_str


def model_transform(model, quantized_param_file, scale_table_file):
    print('Model transform start !')
    scale_table = open(scale_table_file, 'w+')

    activation_scale_str = "activation scale:\nconv_name activation_scale clamp_max\n"
    weight_scale_str = "weight scale:\nconv_name channel_index weight_scale\n"
    params = {}

    index = 0
    act_scale_list = []
    weight_scale_list = []
    m_bits_list = []
    conv_name_list = []
    layer_list = []

    model = model.cpu()
    print(model)

    for name, layer in model.named_modules():
        if isinstance(layer, tflite.Conv2d_quantization):
            print('index = {}, {}'.format(index, name))
            activation_scale_str, weight_scale_str, params, act_scale, weigh_scale, m_bits, conv_name \
                = layer_transform(name, layer, activation_scale_str, weight_scale_str, params)
            # assert False, 'Stop !'
            act_scale_list.append(act_scale)
            weight_scale_list.append(weigh_scale)
            m_bits_list.append(m_bits)
            conv_name_list.append(conv_name)
            layer_list.append(layer)
            index += 1
        elif  isinstance(layer, torch.nn.Linear):
            weight_name = '{} weight'.format(name)
            params[weight_name] = layer.weight.detach().numpy().astype(float)
            bias_name = '{} bias'.format(name)
            params[bias_name] = layer.bias.detach().numpy().astype(float)
            
    #if not layer.power_of_two_scale:
    multiplier_scale_str = "multiplier for conv:\nconv_name channel_index M_approximation_float M_shift M_multiplier_int\n"
    ''' else:
        multiplier_scale_str = "multiplier for conv:\nconv_name channel_index M_shift \n"
    '''
    multiplier_scale_str = cal_multiplier(model, act_scale_list, weight_scale_list, m_bits_list, conv_name_list, multiplier_scale_str, layer_list)

    # write activation and weight scale
    scale_table.write(activation_scale_str + '\n')
    scale_table.write(weight_scale_str + '\n')

    # write Multiplier for each conv
    scale_table.write(multiplier_scale_str)

    
    ###save the params of FC 
   
    # save quantized weight and bias
    np.save(quantized_param_file, params)
    print('Model transform end !')

def gen_cout(model): 
    cout_file= open('data/cout.txt', 'w+')

    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d,torch.nn.Linear)):
            Cout=layer.weight.shape[0]
            cout_file.write(name+" "+str(Cout)+"\n")

    cout_file.close()
    print('gen_cout end !')

def gen_quant_scale_of_output(model, quant_scale_of_output_file): 
    f = open(quant_scale_of_output_file, 'w')

    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d,torch.nn.Linear)):
            if layer.classify_layer:
                #  the reference inference code of "pred_scale" can be found in line 830 of tflite_quantization_PACT_weight_and_act.py
                clamp_range = c_round(2 ** (c_round(layer.weight_bits) - 1) - 1)
                pred_scale = layer.pred_max / clamp_range

                f.write(name+" "+str(pred_scale)+"\n")

    f.close()
    print('gen_quant_scale_of_output end !')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model transform 
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--model_path", type=str, default='data/int8_best.pt', help='quantized model path') 
    parser.add_argument("--result_name", type=str, default='data/output_int8', help='results path of transformed_model')
    
    parser.add_argument('-C', '--config', required=True)
    parser.add_argument('-M', '--mode', default='eval')
    parser.add_argument('-F', '--flops', required=True)
    parser.add_argument('-D', '--data', required=True)
    parser.add_argument('--chcfg', default=None)  
    
    parser.add_argument('--distributed', action='store_true', default=True,
                        help='disables CUDA training')###########
    parser.add_argument('--gpu',  default=None,type=str)###############
    
    args = parser.parse_args()
    
    opt=args
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:63637',
                                world_size=1, rank=0)
        
    #args = tools.get_args(parser)
    config = tools.get_config(args) 
    
    tools.init(config)
    tb_logger, logger = tools.get_logger(config)
    
    #tools.check_dist_init(config, logger)
    checkpoint = tools.get_checkpoint(config)
    runner = tools.get_model(args,config, checkpoint)
    # loaders = tools.get_data_loader(config)
    
   
    if 1:
        model=runner.model
        model=model.module 
        if 1:#opt.quantization:  
            model = model.cpu() 
            model = tflite.replace(model=model, inference_type="full_int", Mn_aware=True,fuseBN=True) 
       
        if 1:#opt.Mn_aware:
            tflite.update_next_act_scale(model)
        
        #model=setDDP(model,args)  
    
        if 1:#opt.quantization and opt.fuseBN is not None:  
            print('Fusing BN and Conv2d_quantization layers... ')
            count = 0
            for name,m in model.named_modules(): #model.module.named_modules():
                print(name)
                if type(m) in [AdaptiveBasicBlock]:
                    m.conv1 = tflite.fuse_conv_and_bn(m.conv1, m.bn1)  # update conv
                    m.conv1.Mn_aware=True
                    delattr(m, 'bn1')  # remove batchnorm
                    
                    m.conv2 = tflite.fuse_conv_and_bn(m.conv2, m.bn2)  # update conv
                    m.conv2.Mn_aware=True
                    delattr(m, 'bn2')  # remove batchnorm
                     
                    if m.downsample is not None:
                        conv= tflite.fuse_conv_and_bn(m.downsample[0], m.downsample[1])  # update conv
                        conv.Mn_aware=True
                        m.downsample = nn.Sequential(conv)
                        
                    m.forward = m.fuseforward  # update forward#### 
                    if hasattr(m.conv1,'Mn_aware') and m.conv1.Mn_aware:
                        m.register_buffer('block_M', torch.ones(1))
                        m.register_buffer('block_n', torch.zeros(1))
                        print('have block_M..............................')
                    count += 1
            model.conv1 = tflite.fuse_conv_and_bn(model.conv1, model.bn1)  # update conv
            model.conv1.Mn_aware=True
            delattr(model, 'bn1')  # remove batchnorm
            model.forward=model.fuseforward #####dDDP
        if opt.model_path is not None:
            state_dict = torch.load(opt.model_path)['model']#.float().state_dict()
            state_dict1={key[7:]:state_dict[key] for key in state_dict.keys()} ##去掉DDP的'.module'
            model.load_state_dict(state_dict1, strict=False) 
            tflite.update_next_act_scale(model)
            # replace alpha
            #tflite.replace_alpha(model, bit=opt.quantization_bits, check_multiplier=True)

    # load quantized mdoel
    #model = torch.load(opt.model_path, map_location=torch.device('cpu'))
    tflite.layer_transform('full_int', model)  ####EMA fuseBN-->layer.bias/layer.weight 

    ####### 20240129 新增代码 ##############
    for name, layer in model.named_modules():
        if isinstance(layer, (tflite.Conv2d_quantization)):
            layer.inference_type = "all_fp"

    temp_input = torch.zeros([1, 3, 224, 224]).cuda()
    model = model.eval().cuda()
    # # update alpha, act_scale, weight_sacle, M, n 
    temp_output = model(temp_input)
    tflite.update_next_act_scale(model)
    del temp_input, temp_output

    for name, layer in model.named_modules():
        if isinstance(layer, (tflite.Conv2d_quantization)):
            layer.inference_type = "full_int"
    ####### 20240129 新增代码 ##############
    
    # transform model
    quantized_param_file = opt.result_name + '_param'
    scale_table_file = opt.result_name + '.scale'
    model_transform(model, quantized_param_file, scale_table_file)
    
    gen_cout(model)
    
    from models.adaptive.resnet import AdaptiveBasicBlock
    
    block_file = opt.result_name +"_blockMn.txt"
    fb=open(block_file,'w')
    for name, layer in model.named_modules():
        if isinstance(layer, AdaptiveBasicBlock):
            out=name+','+str(int(layer.block_n.cpu().numpy()[0]))+','+str(int(layer.block_M.cpu().numpy()[0]))+'\n' 
            fb.write(out)
    fb.close()
    

    # save the quant scale of output from the final layer of each branch
    quant_scale_of_output_file= opt.result_name + '_quant_scale_of_output.txt'
    gen_quant_scale_of_output(model, quant_scale_of_output_file)




