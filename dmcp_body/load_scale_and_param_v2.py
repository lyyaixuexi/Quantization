import argparse
import numpy as np
import os
def load_act_scale_seq(path):
    # 打开文件并读取
    scale_dict ={}
    act_scale_seq=[]
    clamp_max=[]
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
                act_scale_seq.append(float(line.split(" ")[1]))
                clamp_max.append(int(line.split(" ")[2]))


    return act_scale_seq,clamp_max,scale_dict

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

                scale_dict[conv_name]["M_scale"][int(line.split(" ")[1])] = [float(line.split(" ")[2]), int(line.split(" ")[3]), int(line.split(" ")[4])]

    return scale_dict


def load_param(path):

    # 读取param dict
    ori_param_dict = np.load(path, allow_pickle=True).item()

    param_dict = {}

    # 逐一获取key（参数名字）和param（具体的int类型参数）
    for key, param in ori_param_dict.items():
        print(key)
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

def load_cout(fpath):
    fp=open(fpath,'r')
    lines=fp.readlines()
    cout=[]
    for line in lines:
        Co=int(line.split(" ")[1])
        cout.append(Co)
    return cout

def write_int(f,value):
    f.write(value.to_bytes(length=4, byteorder='little', signed=True))
def load_multiplier(path):
    fps=open(path,'r')
    lines=fps.readlines()
    M_n={}
    for line in lines:
        line=line.strip('\n')
        strs=line.split(',')
        key=strs[0]
        #input_max=int(strs[1])
        nchannel=int(strs[1])
        M_n[key]=[]
        #M_n[key].append(input_max)
        M_n[key].append(nchannel)
        print('key={}, nchannels={}'.format(key, nchannel))
        M_ns=[]
        for i in range(0,nchannel*2,2):
            n=int(strs[2+i])
            M=int(strs[2+i+1])
            M_n[key].append(n)
            M_n[key].append(M)
    fps.close()
    
    print(M_n) 
    return M_n

def load_block_Mns(block_Mn_file):
    #load block_Mn.txt: block名，n,M
    
    fm=open(block_Mn_file,'r')
    lines=fm.readlines()
    print(lines)
    block_Mns={}
    for line in lines:
        line=line.strip('\n')
        strs=line.split(',')
        
        key=strs[0] 
        nc=int(len(strs)/2)
        print('key={},n_blockMn={}'.format(key,nc))
        block_Mns[key]=[]
        for i in range(1,len(strs),2):
            n=int(strs[i])
            M=int(strs[i+1])
            block_Mns[key].append(n)
            block_Mns[key].append(M)
    fm.close()
    print(block_Mns)
    return block_Mns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 传入scale和param的路径
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--scale_file", type=str, default='data/output_int16_p035.scale')
    parser.add_argument("--param_file", type=str, default='data/output_int16_p035_param.npy')
    parser.add_argument("--cout_file", type=str, default='data/cout.txt')
    parser.add_argument("--save_dir", type=str, default='data')
    parser.add_argument("--block_Mn_file", type=str, default='data/output_int16_block_Mn.txt')

    opt = parser.parse_args()
    # load scale and weight
    scale_dict = load_scale(opt.scale_file)
    param_dict = load_param(opt.param_file)
    
    bits = opt.bits
    key_nums = len(scale_dict.keys()) #53 包含fc
    print("key_nums: ",key_nums)
    out_path=os.path.join(opt.save_dir,"multiplier_int{}.bin".format(bits))
    f = open(out_path,"wb")
    write_int(f, int(bits))
    write_int(f,int(key_nums))
    '''
    input_max = np.zeros((key_nums), dtype = np.int)
    #输出input_max[53]，用于input的截断 
    
    i = 0
    for key, param in scale_dict.items():
        input_max[i] = scale_dict[key]["activation_scale"][1]
        i += 1

    for i in range(key_nums):
        write_int(f,int(input_max[i]))
    '''
    #输出multiplier，n
    multiplier = []
    num = 0
    outs=[]
    out_s_path=os.path.join(opt.save_dir,"multiplier_int{}.txt".format(bits))
    fp_s=open(out_s_path,"w")
    for key, param in scale_dict.items():
        #multiplier_int{}.txt文件格式：layer_name, 本层channels数，本层各个channel的<n,M>
        out=key+","
        #out+=str(scale_dict[key]["activation_scale"][1])+","  #根据新的方案，不需要再对输入进行截断,所以不需要input_max 
        
        length = len(scale_dict[key]['M_scale'])
        out+=str(length)+","
        num += length
        multiplier0 = np.zeros((length,2), dtype = np.int)
        for i in range(length):
            multiplier0[i][0] = scale_dict[key]["M_scale"][i][1]
            multiplier0[i][1] = scale_dict[key]["M_scale"][i][2]
            out+=str(multiplier0[i][0])+","+str(multiplier0[i][1])+","
            
            if(multiplier0[i][1]>4095):
                print(key," %d channel overflow: %d"%(i,multiplier0[i][1]))
        out+="\n"
        fp_s.write(out)
        multiplier.append(multiplier0)
    fp_s.close()
    #test
    load_multiplier(out_s_path)
    
    
    write_int(f,int(num)) ####number of channels:M/n
    for i in range(key_nums):
        for j in range(len(multiplier[i])):
            write_int(f,int(multiplier[i][j][0]))
            write_int(f,int(multiplier[i][j][1]))

    cout=load_cout(opt.cout_file)
    for co in cout:
        write_int(f,co)
    f.close()
    
    
    #load block_Mn.txt 
    load_block_Mns(opt.block_Mn_file)
    
            
        
        
