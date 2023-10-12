import argparse
import logging

import torch
import torchvision
import numpy as np

import tflite_quantization_PACT_weight_and_act as tflite

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
    min = - 2 ** (bit - 1) + 1
    max = 2 ** (bit - 1) - 1
    # print(min, max)
    # print('calculate_sign_overflow_number', torch.min(x), torch.max(x))
    down_overflow_index = x < min
    up_overflow_index = x > max
    No = torch.sum(down_overflow_index) + torch.sum(up_overflow_index)
    return No

def layer_transform(name, layer, activation_scale_str, weight_scale_str, params):
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
    params[weight_name] = quantized_weight.numpy().astype(int)

    # process bias
    print('process bias')
    quantized_bias = c_round(bias)
    # Test bias
    No = calculate_sign_overflow_number(quantized_bias, bias_bits)
    print(quantized_bias.abs().max())
    print(2 ** (bias_bits - 1) - 1)
    if No != 0:
        print(No)
        assert False, "quantized bias error !!!"

    bias_name = '{} bias'.format(name)
    params[bias_name] = quantized_bias.numpy().astype(int)

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
    for name, layer in model.named_modules():
        # if isinstance(layer, tflite.Conv2d_quantization) or isinstance(layer, tflite.Linear_quantization):
        if isinstance(layer, tflite.Conv2d_quantization):

            if not layer.power_of_two_scale:
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

        # elif isinstance(layer, tflite.Linear_quantization):
        #     print('index = {}, {}'.format(index, name))
        #     activation_scale_str, weight_scale_str, params, act_scale, weigh_scale, m_bits, conv_name \
        #         = layer_transform_linear(name, layer, activation_scale_str, weight_scale_str, params)
        #     act_scale_list.append(act_scale)
        #     weight_scale_list.append(weigh_scale)
        #     m_bits_list.append(m_bits)
        #     conv_name_list.append(conv_name)
        #     layer_list.append(layer)
        #     index += 1

    if not layer.power_of_two_scale:
        multiplier_scale_str = "multiplier for conv:\nconv_name channel_index M_approximation_float M_shift M_multiplier_int\n"
    else:
        multiplier_scale_str = "multiplier for conv:\nconv_name channel_index M_shift \n"

    multiplier_scale_str = cal_multiplier(model, act_scale_list, weight_scale_list, m_bits_list, conv_name_list, multiplier_scale_str, layer_list)

    # write activation and weight scale
    scale_table.write(activation_scale_str + '\n')
    scale_table.write(weight_scale_str + '\n')

    # write Multiplier for each conv
    scale_table.write(multiplier_scale_str)

    # save quantized weight and bias
    np.save(quantized_param_file, params)
    print('Model transform end !')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model transform
    # parser.add_argument("--model_path", type=str, default='best_int14.pt', help='quantized model path')
    # parser.add_argument("--result_path", type=str, default='output_int14.txt', help='results path of transformed_model')
    parser.add_argument("--model_path", type=str, default='data/int8_best.pt', help='quantized model path')
    parser.add_argument("--result_name", type=str, default='data/output_int8', help='results path of transformed_model')
    opt = parser.parse_args()

    # load quantized mdoel
    model = torch.load(opt.model_path, map_location=torch.device('cpu'))
    # transform model
    quantized_param_file = opt.result_name + '_param'
    scale_table_file = opt.result_name + '.scale'
    model_transform(model['model'], quantized_param_file, scale_table_file)




