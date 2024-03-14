# -*- coding:utf-8  -*-

import argparse
import utils.distributed as dist
import utils.tools as tools
import torch
import torch.nn as nn

import torch.distributed as dist
import torch.multiprocessing as mp

import torch.backends.cudnn as cudnn

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
    #args = tools.get_args(parser)
    config = tools.get_config(args)
    tools.init(config)
    tb_logger, logger = tools.get_logger(config)
    
    #tools.check_dist_init(config, logger)

    checkpoint = tools.get_checkpoint(config)
    runner = tools.get_model(args,config, checkpoint)
    loaders = tools.get_data_loader(config)

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

    if dist.is_master():
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
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:63637', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
    parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
    
    args = parser.parse_args()
    
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
