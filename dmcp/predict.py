
import argparse
import utils.distributed as dist
import utils.tools as tools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import torch.distributed as dist
import torch.multiprocessing as mp

import torch.backends.cudnn as cudnn
import tflite_quantization_PACT_weight_and_act as tflite
from overflow_utils import *

from models.adaptive.resnet import AdaptiveBasicBlock 
from rwfile  import *
import cv2
import torchvision.transforms as transforms
import PIL.Image as Image
import copy
 


def c_round(n):
    return torch.where(n>0.0, torch.floor(n+0.5), torch.ceil(n-0.5))

def register_hook(model, func):
    handle_list = []
    for name, layer in model.named_modules():
        if isinstance(layer, tflite.Conv2d_quantization):
            handle = layer.register_forward_hook(func)
            handle_list.append(handle)
    return handle_list


conv_layer_input_list = []
conv_layer_output_list = []


def hook_conv_results(module, input, output):
    conv_layer_input_list.append(input[0].detach().clone())
    conv_layer_output_list.append(output.detach().clone())


def setDDP(model,args):
    if args.distributed: 
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu) 
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu],find_unused_parameters=True)
        else:
            model.cuda() 
            model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else: 
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    return model

def convert_to_full_int(net):
        # inference one time to update M and n
        net.eval()
         
        #tflite.update_next_act_scale(model) 
        # 　replace float param to interger param
        index = 0
        act_scale_list = [] 
        for name, layer in net.named_modules():
            if isinstance(layer, (tflite.Conv2d_quantization)):
                act_scale_list.append(layer.act_scale) 
                layer.inference_type = "full_int"
                layer.index = index
                index += 1 
                # print some information
                print("act_bits:{}  weight_bits:{} bias_bits:{}".format(layer.act_bits, layer.weight_bits,
                                                                    layer.bias_bits))

        input_scale = tflite.layer_transform("full_int", net)
        print('input_scale={}'.format(input_scale))

        return act_scale_list, input_scale
    
def evaluate(runner, loaders):
    train_loader, val_loader = loaders
    runner.infer(val_loader, train_loader=train_loader)
if __name__ == '__main__':
    #TORCH_DISTRIBUTED_DEBUG=DETAIL
    parser = argparse.ArgumentParser(description='DMCP Implementation')
    
    parser.add_argument('-C', '--config', required=True)
    parser.add_argument('-M', '--mode', default='eval')
    parser.add_argument('-F', '--flops', required=True)
    parser.add_argument('-D', '--data', required=True)
    parser.add_argument('--chcfg', default=None)
    parser.add_argument('--bits',type=int,default=4,help="img file path")  # weight/darknet53_448.weights
    parser.add_argument('--img_path',type=str,default="",help="img file path")  # weight/darknet53_448.weights
    
    parser.add_argument('--model_path',type=str,default="",help="weight file path")  # weight/darknet53_448.weights
    parser.add_argument('--raw',action='store_true',default=False)
    
    parser.add_argument('--distributed', action='store_true', default=True,
                        help='disables CUDA training')###########
    parser.add_argument('--gpu',  default=None,type=str)###############
    
    args = parser.parse_args()
    ############################
    setbits(args.bits)
    removeoutfile()
    # tflite.DEBUG_Q=1
    tflite.DEBUG_Q=0
    ############################
    
    opt=args
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:63634',
                                world_size=1, rank=0)
        
    #args = tools.get_args(parser)
    config = tools.get_config(args) 
    
    tools.init(config)
    tb_logger, logger = tools.get_logger(config)
    
    #tools.check_dist_init(config, logger)
    checkpoint = tools.get_checkpoint(config)
    runner = tools.get_model(args,config, checkpoint)
    
   
    if args.raw:  
        #if args.mode == 'evaluate':
        #elif args.mode == 'calc_flops':
        flops = tools.get_model_flops(config, runner.get_model())
        print('flops: {}'.format(flops))
        #elif args.mode == 'calc_params': 
        params = tools.get_model_parameters(runner.get_model())
        print('params: {}'.format(params)) 
        loaders = tools.get_data_loader(config)
        evaluate(runner, loaders)
    else:
        model=runner.model
        model=model.module 
        if 1:#opt.quantization:  
            model = model.cpu() 
            model = tflite.replace(model=model, inference_type="all_fp", Mn_aware=True,fuseBN=True) 
        
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
            # state_dict = torch.load(opt.model_path)['model']#.float().state_dict()
            state_dict = torch.load(opt.model_path)['model']#.float().state_dict()
            state_dict1={key[7:]:state_dict[key] for key in state_dict.keys()} ##去掉DDP的'.module'
            model.load_state_dict(state_dict1, strict=False) 
            tflite.update_next_act_scale(model)
            # replace alpha
            #tflite.replace_alpha(model, bit=opt.quantization_bits, check_multiplier=True)
    
        print(model) 

        for name,module in model.named_modules(): 
           if isinstance(module,tflite.Conv2d_quantization):
                module.out_clamp_range=torch.tensor([tflite.c_round(2 ** (tflite.c_round(module.weight_bits) - 1) - 1)])


        convert_to_full_int(model)
        model.eval()
        
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

        # register hook to get conv_layer‘s input and output
        register_hook(model, hook_conv_results)

        # runner.model.module=model
        runner.model=model.cuda()
        #if args.mode == 'evaluate':
        #evaluate(runner, loaders)
        # pic = cv2.imread(args.img_path)
        # pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        # pic = cv2.resize(pic, (256, 256))
        # trans=transforms.Compose([
        #     #transforms.Resize(config.augmentation.test_resize),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     #normalize
        # ])
        # print(pic.size)
        # img=Image.fromarray(pic)
        # imgf=trans(img)
        # imgf=imgf.unsqueeze(0).cuda()
        # print(imgf.shape)
        #
        # pred=runner.model(imgf)
        # print(pred)
        # pred=F.softmax(pred)
        # print(pred)
        # class_id=pred.squeeze(0).argmax(dim=0).item()
        # print('class:{}'.format(class_id))


        def get_train_label_list():
            path = './small_sign_classify_class_list.txt'
            labellist = list()
            with open(path, 'r') as f:
                for line in f:
                    i = line.strip('\n')
                    labellist.append(i)
            return labellist


        def transform(img):

            img = img.transpose(2, 0, 1)
            img = img.astype(float) / 256.0

            return img
        def test(model, img):
            model.eval()
            test_loss = 0
            correct = 0
            temp_correct = 0
            img_lst = []
            fail_lst = []
            labels = {}
            img_lst.append(img)
            file_lst = []
            labellist = get_train_label_list()
            for img_path in img_lst:
                temp_label = img_path.split('/')
                try:
                    labels[img_path] = labellist.index(temp_label[-2])
                except:
                    try:
                        labels[img_path] = labellist.index(temp_label[-3])
                    except:
                        try:
                            labels[img_path] = labellist.index(temp_label[-4])
                        except:
                            try:
                                labels[img_path] = labellist.index(temp_label[-5])
                            except:
                                fail_lst.append(img_path)
                                continue
                file_lst.append(img_path)

            img = np.zeros([288, 288, 3])
            label = np.zeros([3, 3], dtype=np.float32)

            ## 这里相当于一拖8，做一个随机拼接图像的操作，完成随机输入
            for i in range(9):
                img_path = file_lst[0]
                temp_img = cv2.imread(img_path)
                # print(temp_img)
                temp_img = cv2.resize(temp_img, (96, 96))

                if i == 0:  # 放中间
                    img[96 * 1:96 * 1 + 96, 96 * 1:96 * 1 + 96, :] = temp_img
                    label[1, 1] = labels[img_path]
                elif i == 1:  # 一号位
                    img[96 * 0:96 * 0 + 96, 96 * 0:96 * 0 + 96, :] = temp_img
                    label[0, 0] = labels[img_path]
                elif i == 2:  # 二号位
                    img[96 * 0:96 * 0 + 96, 96 * 1:96 * 1 + 96, :] = temp_img
                    label[0, 1] = labels[img_path]
                elif i == 3:  #
                    img[96 * 0:96 * 0 + 96, 96 * 2:96 * 2 + 96, :] = temp_img
                    label[0, 2] = labels[img_path]
                elif i == 4:
                    img[96 * 1:96 * 1 + 96, 96 * 0:96 * 0 + 96, :] = temp_img
                    label[1, 0] = labels[img_path]
                elif i == 5:
                    img[96 * 1:96 * 1 + 96, 96 * 2:96 * 2 + 96, :] = temp_img
                    label[1, 2] = labels[img_path]
                elif i == 6:
                    img[96 * 2:96 * 2 + 96, 96 * 0:96 * 0 + 96, :] = temp_img
                    label[2, 0] = labels[img_path]
                elif i == 7:
                    img[96 * 2:96 * 2 + 96, 96 * 1:96 * 1 + 96, :] = temp_img
                    label[2, 1] = labels[img_path]
                elif i == 8:
                    img[96 * 2:96 * 2 + 96, 96 * 2:96 * 2 + 96, :] = temp_img
                    label[2, 2] = labels[img_path]

            img = cv2.resize(img,(224, 224))
            
            # save 3*3 formatimage
            cv2.imwrite("test_input_image.jpg", img)
            img = cv2.imread("test_input_image.jpg")

            img = transform(img)
            img = torch.from_numpy(img).float().cuda()
            label = torch.from_numpy(label).float().cuda()

            data = img.unsqueeze(0)

            print("(data.size():{}".format(data.size()))


            output = model(data)

            # print("data after convert to float==>")
            # print(output)
            print(output.dtype)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()

            return img, label, pred, correct


        # path = '/mnt/cephfs/home/lyy/Quantization/dmcp/pl10/0###SZ_1_1_17_20170613_093948_3ae8f77b.pickle_04700_655_190_698_233.jpg_003030_YUV_bt601V.jpg'
        path = args.img_path
        print("test image:{}".format(path))
        img, label, pred, correct = test(runner.model, path)
    
        print("label", label)
        print("pred", pred)
        print("correct", correct)


        ############ for debug ##################
        conv_layer_input_list_1 = copy.deepcopy(conv_layer_input_list)
        conv_layer_output_list_1 = copy.deepcopy(conv_layer_output_list)
        conv_layer_input_list.clear()
        conv_layer_output_list.clear()

        act_scale_list = []
        next_act_list = []
        for name, layer in model.named_modules():
            if isinstance(layer, (tflite.Conv2d_quantization)):
                act_scale_list.append(layer.act_scale)
                next_act_list.append(layer.next_act_scale) 
                layer.inference_type = "all_fp"

        img, label, temp_pred, temp_correct = test(runner.model, path)
    
        print("label", label)
        print("temp_pred", temp_pred)
        print("temp_correct", temp_correct)
    
        conv_layer_input_list_2 = copy.deepcopy(conv_layer_input_list)
        conv_layer_output_list_2 = copy.deepcopy(conv_layer_output_list)
        conv_layer_input_list.clear()
        conv_layer_output_list.clear()

        print("input_number: {}".format(len(conv_layer_input_list_1)))
        print("input_number: {}".format(len(conv_layer_input_list_2)))
        print("output_number: {}".format(len(conv_layer_output_list_1)))
        print("output_number: {}".format(len(conv_layer_output_list_2)))

        for i in range(len(act_scale_list)):
            if i in [0]:
                input_error = 0
            else:
                input_error = (c_round(conv_layer_input_list_2[i] / act_scale_list[i]) - c_round(conv_layer_input_list_1[i])).abs().mean() / c_round(conv_layer_input_list_1[i]).abs().mean()
            
            if i in [20]:
                output_error = (conv_layer_output_list_2[i] - conv_layer_output_list_1[i]).abs().mean() / conv_layer_output_list_1[i].abs().mean()
            else:
                output_error = (c_round(conv_layer_output_list_2[i] / next_act_list[i]) - c_round(conv_layer_output_list_1[i])).abs().mean() / \
                        c_round(conv_layer_output_list_1[i]).abs().mean()
            print("layer:{} act_scale:{} input_error:{} output_error:{}".format(i, act_scale_list[i], input_error, output_error))

        ############ for debug ##################


    
        #elif args.mode == 'calc_flops':
        #flops = tools.get_model_flops(config, runner.get_model())
        #print('flops: {}'.format(flops))
        #elif args.mode == 'calc_params': 
        #params = tools.get_model_parameters(runner.get_model())
        #print('params: {}'.format(params))
        
    