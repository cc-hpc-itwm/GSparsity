import argparse
import os
# os.environ["NCCL_DEBUG"] = "INFO"
import random
import shutil
import time
import warnings
import logging
import numpy as np
import glob
import sys
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from ProxSGD_for_filters import ProxSGD

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://192.168.152.41:23456', type=str,
#                     help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', action='store_true', default=True,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--path_to_save', type=str, default="imagenet", help='path to the folder where the experiment will be saved')
parser.add_argument('--run_id', type=str, default=None, help='the identifier of this specific run')


best_acc1 = 0


def main():
    args = parser.parse_args()

    args.dist_url = 'tcp://{}:23456'.format(os.environ['CARME_MASTER_IP'])
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    
    global best_acc1
    args.gpu = gpu

    
    logger = set_logger(logger_name="{}/_log.txt".format(args.resume))    
    logger.info('CARME Slurm ID: {}'.format(os.environ['SLURM_JOBID']))
    logger.info("args = %s", args)
    
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
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

#     print("torch.cuda.is_available is {}, args.distributed is {}, args.gpu is {}, ".format(torch.cuda.is_available(), args.distributed, args.gpu))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            print("torch.nn.DataParallel(model).cuda()")
            model = torch.nn.DataParallel(model).cuda()

    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    # optionally resume from a checkpoint
    if args.resume:
        path_to_checkpoint = "{}/checkpoint.pth.tar".format(args.resume)
        if os.path.isfile(path_to_checkpoint):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(path_to_checkpoint)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(path_to_checkpoint, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    num_zero_filters, num_total_filters, num_zero_params, num_total_params = print_nonzeros_filters(model, logger, pruning_threshold=1e-6)
    
    compute_and_save_mask(model, "{}/mask".format(args.resume), pruning_threshold=1e-6)



def set_logger(logger_name, level=logging.INFO):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    log_format = logging.Formatter("%(asctime)s %(message)s", '%Y-%m-%d %H:%M:%S')

    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def print_nonzeros_filters(model, logger, pruning_threshold=1e-6):
    num_zero_filters = 0
    num_total_filters = 0    
    
    num_zero_filters_per_layer = [0, 0, 0, 0]
    num_total_filters_per_layer = [0, 0, 0, 0]
    
    num_zero_params = 0
    num_total_params = 0
    
    for name, param in model.named_parameters():
        num_total_params += param.numel()
        if "layer" in name and "conv" in name:
            layer = int(name[12])
            filter_norms = torch.norm(param.view(param.shape[0],-1),p=2,dim=1)
#             print("name: {}, number of output channels {}".format(name, filter_norms.size()))
#             print("    value {}".format(filter_norms))
            num_zero_filters_tmp = (torch.sum(filter_norms <= pruning_threshold)).item()
            num_total_filters_tmp = param.shape[0]
            num_zero_filters += num_zero_filters_tmp
            num_total_filters += num_total_filters_tmp
            
            num_zero_params += param.numel() * num_zero_filters_tmp / num_total_filters_tmp
            
            num_zero_filters_per_layer[layer - 1]  +=  num_zero_filters_tmp
            num_total_filters_per_layer[layer - 1] += num_total_filters_tmp
        elif "layer" in name and "bn" in name:
            num_zero_params += param.numel() * num_zero_filters_tmp / num_total_filters_tmp

    logger.info("pruning threshold: {}, zero/total filters (ratio): {}/{} ({})".format(pruning_threshold, num_zero_filters, num_total_filters, num_zero_filters/num_total_filters))
    logger.info("pruning threshold: {},  zero/total params (ratio): {}/{}M ({})".format(pruning_threshold, num_zero_params/1e6, num_total_params/1e6, num_zero_params/num_total_params))
    
    for layer in range(0, 4):
        logger.info("pruning threshold: {}, layer {}: zero/total filters (conv1/2/3) {}/{} ({}%)".format(pruning_threshold, layer, num_zero_filters_per_layer[layer], num_total_filters_per_layer[layer], num_zero_filters_per_layer[layer]/num_total_filters_per_layer[layer]*100))
    logger.info("pruning threshold: {},  total: zero/total filters  (conv1/2/3) {}/{} ({}%)".format(pruning_threshold, np.sum(num_zero_filters_per_layer), np.sum(num_total_filters_per_layer), np.sum(num_zero_filters_per_layer)/np.sum(num_total_filters_per_layer)*100))
    
    return num_zero_filters, num_total_filters, num_zero_params, num_total_params


def compute_and_save_mask(model, file_path, pruning_threshold=1e-6):
    # valid for resnet50 only
    mask = [[[[],[],[]], [[],[],[]], [[],[],[]]], 
            [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]],
            [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]],
            [[[],[],[]], [[],[],[]], [[],[],[]]]]
    for name, param in model.named_parameters():
#         print("name {}, size {}".format(name, param.size()))
        if "layer" in name:
            layer_index = int(name[12])
            block_index = int(name[14])
            if "conv" in name:
                conv_index = int(name[20])
                filter_norms = torch.norm(param.view(param.shape[0], -1), p=2, dim=1)
                mask[layer_index-1][block_index][conv_index-1] = (filter_norms.gt(pruning_threshold)).cpu().detach().numpy()

#     for layer_index, mask_layer in enumerate(mask):
#         for block_index, mask_block in enumerate(mask_layer):
#             for conv_index, mask_conv in enumerate(mask_block):
#                 print("  layer_index is {},{},    {}".format(layer_index+1, block_index, conv_index+1))
#                 print(mask_conv)

    np.save(file_path, mask)

if __name__ == '__main__':
    main()