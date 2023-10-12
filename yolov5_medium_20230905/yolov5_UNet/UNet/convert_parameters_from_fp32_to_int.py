import argparse
import numpy as np
import os

def load_param(path):

    # 读取param dict
    ori_param_dict = np.load(path, allow_pickle=True).item()

    param_dict = {}

    # 逐一获取key（参数名字）和param（具体的int类型参数）
    for key, param in ori_param_dict.items():

        conv_name = key.split(" ")[0]

        if not conv_name in param_dict:
            param_dict[conv_name] = {}

        param_type = key.split(" ")[1].strip()
        param_dict[conv_name][param_type] = param

    for key, param in param_dict.items():
        print(key)
        output_channel_number = param_dict[key]["bias"].shape[0]
        param_dict[key]["output_channel_number"] = output_channel_number

    return param_dict


def detect_bit_number(paramer):
    bit32_max = 2 ** 31 - 1
    bit16_max = 2 ** 15 - 1
    bit8_max = 2 ** 7 - 1
    bit4_max = 2 ** 3 - 1
    max_paramer = np.max(np.abs(paramer))

    if max_paramer <= bit4_max:
        return 4
    elif max_paramer <= bit8_max:
        return 8
    elif max_paramer <= bit16_max:
        return 16
    elif max_paramer <= bit32_max:
        return 32
    else:
        return max_paramer

def convert_4bit_param_from_int8_to_uint8(param): # 主要是处理符号位。int8的符号位是在从左往右第1位，转成uint8后，要把符号位挪到从左往右第5位
    param = param.astype(np.uint8)
    param = np.where(param<128, param, 255-param+1+8)
    return param


def convert_4bit_param_from_uint8_to_int8(param): # 主要是处理符号位。int8的符号位是在从左往右第1位，转成uint8后，要把符号位挪到从左往右第5位
    param = np.where(param<8, param, 255-(param-8-1))
    param = param.astype(np.int8)
    return param


def convert_int64_to_lowbitint(key, param):
    bit_number = detect_bit_number(paramer=param)
    key = key + " " + str(bit_number)

    if bit_number == 32:
        param = param.astype(np.int32)
    elif bit_number == 16:
        param = param.astype(np.int16)
    elif bit_number == 8:
        param = param.astype(np.int8)
    elif bit_number == 4:
        param_shape = param.shape
        param_flatten = param.flatten()
        param_number = param_flatten.shape[0]

        param_flatten = convert_4bit_param_from_int8_to_uint8(param_flatten)

        if param_number % 2 == 0:
            concat_param = param_flatten[:param_number//2] + param_flatten[param_number//2:]*16
        else:
            print(param_shape)
            concat_param = param_flatten[:(param_number+1)//2] + np.concatenate((param_flatten[(param_number+1)//2:]*16, np.array([0])), axis=0)

        key = key + " "+"_".join([str(x) for x in list(param_shape)])
        param = concat_param

    else:
        print("Unsupport bit number: {}".format(bit_number))
        assert False

    return key, param


# todo
def parase_lowbitint(key, param):
    key_split_list = key.split(" ")
    key = " ".join(key_split_list[:2])

    if key_split_list[2] == "4":
        original_shape = [int(x) for x in key_split_list[3].split("_")]

        param_number = 1
        for i in original_shape:
            param_number *= i

        if param_number % 2 == 0:
            param_flatten = np.concatenate((param % 16, param // 16), axis=0)
            param_flatten = convert_4bit_param_from_uint8_to_int8(param_flatten)

        else:
            param_flatten = np.concatenate((param % 16, param // 16), axis=0)[:-1]
            param_flatten = convert_4bit_param_from_uint8_to_int8(param_flatten)

        param = np.resize(param_flatten, original_shape)

    return key, param

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 传入fp32格式的param.npy路径
    parser.add_argument("--fp32_quantized_param_file", type=str, default='data/Unet_output_int4_half_channel_strict_param.npy')

    opt = parser.parse_args()

    compress_quantized_param_file = os.path.splitext(opt.fp32_quantized_param_file)[0]+"_compress.npy"

    # 读取原始param dict 并压缩参数，原始param.npy使用int64存储参数
    param_dict = np.load(opt.fp32_quantized_param_file, allow_pickle=True).item()
    compress_param_dict = {}
    for key, param in param_dict.items():
        compress_key, compress_param = convert_int64_to_lowbitint(key, param) # format of key: "conv_name weight/bias bit_number"
        compress_param_dict[compress_key] = compress_param

    np.save(compress_quantized_param_file, compress_param_dict)

    print("save param to: {}".format(compress_quantized_param_file))

    # 读取compress param dict 并还原参数
    compress_param_dict = np.load(compress_quantized_param_file, allow_pickle=True).item()
    recover_param_dict = {}
    for key, param in compress_param_dict.items():
        recover_key, recover_param = parase_lowbitint(key, param) # format of key: "conv_name weight/bias"
        recover_param_dict[recover_key] = recover_param

        # compare
        check_the_same = (recover_param_dict[recover_key] == param_dict[recover_key]).all()
        print("verify key: {} is the same: {}".format(recover_key, check_the_same))

    compressed_model_size = os.path.getsize(compress_quantized_param_file)/1024/1024  # convert number of Byte to MB
    print("save param to: {}  the size is:{:.2f} MB".format(compress_quantized_param_file, compressed_model_size))
