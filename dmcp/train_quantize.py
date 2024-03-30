# -*- coding:utf-8  -*-
import os 
import argparse
import utils.distributed as dist
import utils.tools as tools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torch.distributed as dist
import torch.multiprocessing as mp

import torch.backends.cudnn as cudnn
import tflite_quantization_PACT_weight_and_act as tflite
from overflow_utils import *

from models.adaptive.resnet import AdaptiveBasicBlock 
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
def train(config, runner, loaders, checkpoint, tb_logger):
    # load optimizer and scheduler
    optimizer = tools.get_optimizer(runner.get_model(), config, checkpoint)
    if config.get('arch_lr_scheduler', False):
        assert len(optimizer) == 2

        lr_scheduler = tools.get_lr_scheduler(optimizer[0], config.lr_scheduler)
        arch_lr_scheduler = tools.get_lr_scheduler(optimizer[1], config.arch_lr_scheduler)
        lr_scheduler = (lr_scheduler, arch_lr_scheduler)
    else:
        lr_scheduler = tools.get_lr_scheduler(optimizer, config.lr_scheduler)

    # train and calibrate
    train_loader, val_loader = loaders
    runner.train(train_loader, val_loader, optimizer, lr_scheduler, tb_logger)
    runner.infer(val_loader, train_loader=train_loader)


def evaluate(runner, loaders):
    train_loader, val_loader = loaders
    runner.infer(val_loader, train_loader=train_loader)


    
def main(args):
    opt=args
    
    #args = tools.get_args(parser)
    config = tools.get_config(args)
    config.save_path=args.save_dir
    
    tools.init(config)
    tb_logger, logger = tools.get_logger(config)
    
    #tools.check_dist_init(config, logger)

    checkpoint = tools.get_checkpoint(config)
    runner = tools.get_model(args,config, checkpoint)
    loaders = tools.get_data_loader(config)

    args.lr=config.lr_scheduler.base_lr
    
    
    ######
    model=runner.model#.module 有DDP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_path != '':
        #------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        #------------------------------------------------------#
        print('Load weights {}.'.format(args.model_path))
        
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict=torch.load(args.model_path, map_location = device)['model']
         
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}###################
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    model=model.module 
    if opt.quantization:  
        model = model.cpu() 
        model = tflite.replace(model=model, quantization_bits=opt.quantization_bits,
                                   m_bits=opt.m_bits, bias_bits=opt.bias_bits, inference_type="all_fp", Mn_aware=opt.Mn_aware,fuseBN=True if opt.fuseBN is not None else False)#Mn_aware=True) 
     
    if opt.Mn_aware:
        tflite.update_next_act_scale(model)
        #tflite.replace_next_act_scale(model)  
        
    #setDDP(model,args)#################### 
    model=setDDP(model,args)
    #model = model.cuda(args.gpu)
            
    if opt.quantization and opt.fuseBN is not None:
            # load pretrain quantized model
            model_dict      = model.state_dict()
            pretrained_state_dict = torch.load(opt.fuseBN)['model']
            print('model_dict:{}'.format(model_dict.keys()))
            print('pretrained_dict:{}'.format(pretrained_state_dict.keys()))
            pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if np.shape(model_dict[k]) == np.shape(v)}###################
            model.load_state_dict(pretrained_state_dict, strict=False)

            # target_layer_name = "module.layer4.0.conv2.act_scale"
            # target_keyword = 'act_scale'
            # for name, param in pretrained_state_dict.items():
            #     # 检查层的名称是否包含关键字
            #     if target_keyword in name:
            #         # 打印包含关键字的层的参数数值
            #         print(f"Parameters of Layer {name}: {param.data}")

            print('Fusing BN and Conv2d_quantization layers... ')
            count = 0
            for name,m in  model.module.named_modules():
                print(name)
                if type(m) in [AdaptiveBasicBlock]:
                    m.conv1 = tflite.fuse_conv_and_bn(m.conv1, m.bn1)  # update conv
                    delattr(m, 'bn1')  # remove batchnorm
                    
                    m.conv2 = tflite.fuse_conv_and_bn(m.conv2, m.bn2)  # update conv
                    delattr(m, 'bn2')  # remove batchnorm
                     
                    if m.downsample is not None:
                        conv= tflite.fuse_conv_and_bn(m.downsample[0], m.downsample[1])  # update conv
                        m.downsample = nn.Sequential(conv)
                        
                    m.forward = m.fuseforward  # update forward ########################
                    if hasattr(m.conv1,'Mn_aware') and m.conv1.Mn_aware:
                        m.register_buffer('block_M', torch.ones(1))
                        m.register_buffer('block_n', torch.zeros(1))
                        print('have block_M..............................')
                    # print("m.conv.act_max: {}".format(m.conv.act_max))
                    # print("m.conv.min_scale: {}".format(m.conv.min_scale))
                    # print("m.conv.alpha: {}".format(m.conv.alpha))
                    count += 1
            model.module.conv1 = tflite.fuse_conv_and_bn(model.module.conv1, model.module.bn1)  # update conv
            #model.module.conv2 = tflite.fuse_conv_and_bn(model.module.conv2, model.module.bn2)  # update conv
            delattr(model.module, 'bn1')  # remove batchnorm
            #delattr(model.module, 'bn2')  # remove batchnorm
            model.module.forward=model.module.fuseforward #####dDDP 
            

    if opt.load_pretrain_fuseBN is not None:
            pretrained_state_dict = torch.load(opt.load_pretrain_fuseBN)['model']#.float().state_dict()
            model_dict      = model.state_dict()  
            pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if np.shape(model_dict[k]) == np.shape(v)}###################
            
            model.load_state_dict(pretrained_state_dict, strict=False)
            #if opt.Mn_aware is not None:
                  #tflite.replace_next_act_scale(model)
            tflite.update_next_act_scale(model)
            # replace alpha
            #tflite.replace_alpha(model, bit=opt.quantization_bits, check_multiplier=True)
    #print(model.state_dict())
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False  
    #######################################################       
    # count=0
    # for name,module in model.named_modules():
    #        if isinstance(module,tflite.Conv2d_quantization):
    #             if True:#opt.quantization_bits>8:
    #                 print('Conv')
    #                 print(opt.quantization_bits)
    #                 print(module.act_bits.get_device())
    #                 module.act_bits = torch.tensor([opt.quantization_bits],device=module.act_bits.get_device())
    #                 module.weight_bits = torch.tensor([opt.quantization_bits],device=module.act_bits.get_device()) #4bit时的第一层的8it恢复成了4bit
    #
    #                 module.act_num_levels= torch.tensor([tflite.c_round(2 ** module.act_bits)],device=module.act_bits.get_device())
    #                 module.weight_num_levels=torch.tensor([tflite.c_round(2 ** module.weight_bits)],device=module.act_bits.get_device())
    #                 module.out_clamp_range=torch.tensor([tflite.c_round(2 ** (tflite.c_round(module.weight_bits) - 1) - 1)],device=module.act_bits.get_device())
    #                 if  opt.quantization_bits==4:
    #                     module.out_clamp_range=torch.tensor([7],device=module.act_bits.get_device()) #仅用于stage3
    #                 #bias_bits = set_bias_bits
    #                 if  opt.quantization_bits>4 and opt.quantization_bits<=8 :
    #                     module.out_clamp_range=torch.tensor([127],device=module.act_bits.get_device()) #仅用于stage3
    #                 if  opt.quantization_bits>8 and opt.quantization_bits<=16:
    #                     module.out_clamp_range=torch.tensor([32767],device=module.act_bits.get_device()) #仅用于stage3
    #             count+=1
   ####################################################
    #optimizer=optim.Adam(model.parameters(), lr , weight_decay =5e-4)
    
    if args.resume is not None:
        print('load resume {}............'.format(args.resume))
        ckpt=torch.load(args.resume, map_location = device)
        optimizer.load_state_dict(ckpt["optimizer"])
        model.load_state_dict(ckpt["model"])
        start_epoch= ckpt["epoch"]   
         
        print(start_epoch)
    
    
    #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) #######     
    #model.cuda(args.gpu)
    print(model) 
    
    #runner.model.module=model
    runner.model=model
    
    if args.rank==0:
        logger.info(config)

    if args.mode == 'train':
        train(config, runner, loaders, checkpoint, tb_logger)
    elif args.mode == 'evaluate':
        evaluate(runner, loaders)
    elif args.mode == 'calc_flops':
        if dist.is_master():
            flops = tools.get_model_flops(config, runner.get_model())
            logger.info('flops: {}'.format(flops))
    elif args.mode == 'calc_params':
        if dist.is_master():
            params = tools.get_model_parameters(runner.get_model())
            logger.info('params: {}'.format(params))
    else:
        assert checkpoint is not None
        from models.dmcp.utils import sample_model
        sample_model(config, runner.get_model())

    if args.rank==0:
        logger.info('Done')

def main_worker(gpu, ngpus_per_node, args):
    opt=args
    
    args.gpu = gpu
    cudnn.benchmark = True
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    if args.cuda:
        torch.cuda.manual_seed(1)
    
    main(args)
    
if __name__ == '__main__':
    #TORCH_DISTRIBUTED_DEBUG=DETAIL
    parser = argparse.ArgumentParser(description='DMCP Implementation')
    parser.add_argument('-C', '--config', required=True)
    parser.add_argument('-M', '--mode', default='eval')
    parser.add_argument('-F', '--flops', required=True)
    parser.add_argument('-D', '--data', required=True)
    parser.add_argument('--chcfg', default=None)
    parser.add_argument('--model_path',type=str,default="",help="weight file path")  # weight/darknet53_448.weights
    parser.add_argument('--resume',type=str,default=None,help="weight file path")  # weight/darknet53_448.weights
    #main()
    
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='disables CUDA training')###########
    parser.add_argument('--gpu',  default=None,type=str)###############
    #parser.add_argument('--local_rank',  default='0',type=str)###############
    parser.add_argument('--multiprocessing-distributed', default=True,action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:63684', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
    parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
    
    #quantization
    parser.add_argument("--quantization", action='store_true', default=False,help='whether to do training aware quantization')
    parser.add_argument('--quantization_bits', type=int, default=6)
    parser.add_argument('--m_bits', type=int, default=12)
    parser.add_argument('--bias_bits', type=int, default=16)
    parser.add_argument('--use_new_opt', action='store_true', help='use_new_opt')
    parser.add_argument('--fuseBN', type=str, default=None,
                        help='load pretrain quantized model and fuse BatchNormalization Layer and finetune')

     
    parser.add_argument('--load_pretrain_fuseBN', type=str, default=None,
                        help='load pretrain fuseBN quantized model and finetune')
    parser.add_argument('--load_pretrain_and_test', action='store_true',help='load_pretrain_and_test')
    parser.add_argument('--check_overflow', action='store_true',help='check_overflow')
    parser.add_argument('--Mn_aware', action='store_true',default=False,help='Mn_aware')
    
    # overflow aware quantization
    #parser.add_argument('--OAQ_m', type=int, default=50)
    parser.add_argument('--OAQ_m', type=int, default=10) ##############################
    parser.add_argument('--save_dir',type=str,default="weight") 
        
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    torch.manual_seed(1)
    cudnn.deterministic = True
    
    args.cuda = torch.cuda.is_available()
    
    
    print('cuda{}'.format(args.cuda))
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed   #####################
    
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
