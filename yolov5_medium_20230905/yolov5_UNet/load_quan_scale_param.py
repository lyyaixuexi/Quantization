import argparse
import numpy as np

def load_scale(path):
    # 打开文件并读取
    scale_dict = {}
    state = None
    with open(path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()

            if line == "activation scale:":
                state = 1
                continue

            if line == "weight scale:":
                state = 2
                continue

            if line == "multiplier for conv:":
                state = 3
                continue

            if len(line.split(" ")) == 1:
                continue

            if line.split(" ")[0] == 'conv_name':
                continue

            if state == 1:
                conv_name = line.split(" ")[0]
                if not conv_name in scale_dict:
                    scale_dict[conv_name] = {}

                scale_dict[conv_name]["activation_scale"] = [float(line.split(" ")[1]), int(line.split(" ")[2])]

            if state == 2:
                conv_name = line.split(" ")[0]
                if not "weight_scale" in scale_dict[conv_name]:
                    scale_dict[conv_name]["weight_scale"] = {}

                scale_dict[conv_name]["weight_scale"][int(line.split(" ")[1])] = [float(line.split(" ")[2])]

            if state == 3:
                conv_name = line.split(" ")[0]
                if not "M_scale" in scale_dict[conv_name]:
                    scale_dict[conv_name]["M_scale"] = {}

                if len(line.split(" ")) == 5:
                    scale_dict[conv_name]["M_scale"][int(line.split(" ")[1])] = [float(line.split(" ")[2]), int(line.split(" ")[3]), int(line.split(" ")[4])]
                elif len(line.split(" ")) == 3:
                    # for version 2 quantization, only need to load M_shift
                    scale_dict[conv_name]["M_scale"][int(line.split(" ")[1])] = [float(line.split(" ")[2]),]

    return scale_dict


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 传入scale和param的路径
    parser.add_argument("--scale_table_file", type=str, default='int8/output_int8.scale')
    parser.add_argument("--quantized_param_file", type=str, default='int8/output_int8_param.npy')
    opt = parser.parse_args()

    # load scale and weight
    scale_dict = load_scale(opt.scale_table_file)
    param_dict = load_param(opt.quantized_param_file)

    # 查看scale的key
    print("List keys: ")
    for key, param in scale_dict.items():
        print(key) # 卷积层的名字
        print(scale_dict[key].keys()) # dict_keys(['activation_scale', 'weight_scale', 'M_scale'])

    # 查看param和key
    for key, param in param_dict.items():
        print(key) # 卷积层的名字
        print(param_dict[key].keys()) # dict_keys(['weight', 'bias', 'output_channel_number'])


    # 希望看某一量化卷积层的参数：
    conv_name = "model.model.24.m.2"

    print("\n\n ############### scale ###############")

    output_channel_idx = 0

    print('\nscale_dict['+conv_name+']["activation_scale"]')
    print(scale_dict[conv_name]["activation_scale"])  # [float_scale, clamp_max]

    print('\nscale_dict['+conv_name+']["weight_scale"]['+str(output_channel_idx)+']')
    print(scale_dict[conv_name]["weight_scale"][output_channel_idx])  # [float_scale,]

    print('\nscale_dict['+conv_name+']["M_scale"]['+str(output_channel_idx)+']')
    print(scale_dict[conv_name]["M_scale"][output_channel_idx]) # [float_scale, n, M] or [n]


    # 希望看某一卷积层的weiht和bias：
    print("\n ############### param ###############")
    print('\nparam_dict['+conv_name+']["weight"]')
    print(param_dict[conv_name]["weight"])

    print('\nparam_dict['+conv_name+']["bias"]')
    print(param_dict[conv_name]["bias"])

    print('\nparam_dict['+conv_name+']["output_channel_number"]')
    print(param_dict[conv_name]["output_channel_number"])

    print("Good Luck to You!!!")
