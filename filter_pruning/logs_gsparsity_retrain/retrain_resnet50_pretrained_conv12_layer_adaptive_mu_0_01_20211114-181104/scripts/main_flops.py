import argparse
import os
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
torch.autograd.set_detect_anomaly(True)

from ptflops import get_model_complexity_info
import list_of_models as list_of_models
import models_with_reduce_all_conv

# model_names = sorted(name for name in models_with_reduce_all_conv.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models_with_reduce_all_conv.__dict__[name]))

import torchvision.models as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DataDir', default='/home/yangy/Dataset_ImageNet',
                    help='path to dataset')
parser.add_argument('--baseline_model',
                    help='the baseline model to prune and retrain')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--pruning_threshold', default=1e-6, type=float,
                    metavar='W', help='pruning threshold (default: 1e-6)')
parser.add_argument('-p', '--print-freq', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--path_to_resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
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
parser.add_argument('--path_to_save', type=str, default="imagenet-flops", help='path to the folder where the experiment will be saved')
parser.add_argument('--run_id', type=str, default=None, help='the identifier of this specific run')


best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.path_to_resume:
        resume_training = args.path_to_resume
        f = open("{}/run_info.json".format(args.path_to_resume))
        run_info = json.load(f)
        vars(args).update(run_info['args'])
        args.path_to_resume = resume_training
    else:
        if args.run_id is None:
            args.run_id = "{}_{}_lr_{}_momentum_{}_wd_{}_{}".format(args.arch,
                                                                    args.baseline_model,
                                                                    args.lr,
                                                                    args.momentum,
                                                                    args.weight_decay,
                                                                    time.strftime("%Y%m%d-%H%M%S"))
        else:
            args.run_id = "{}_{}_{}_lr_{}_momentum_{}_wd_{}_{}".format(args.arch,
                                                                       args.baseline_model,
                                                                       args.run_id,
                                                                       args.lr,
                                                                       args.momentum,
                                                                       args.weight_decay,
                                                                       time.strftime("%Y%m%d-%H%M%S"))
        args.path_to_save = "{}_{}".format(args.path_to_save, args.run_id)
        create_exp_dir(args.path_to_save) #create_exp_dir(args.path_to_save, scripts_to_save=glob.glob('*.py'))
        
        run_info = {}
        run_info['args'] = vars(args)        
        with open('{}/run_info.json'.format(args.path_to_save), 'w') as f:
            json.dump(run_info, f)
    
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

    
    logger = set_logger(logger_name="{}/_log.txt".format(args.path_to_save))    
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

#     path_to_prune = "{}/mask.npy".format(eval("list_of_models.%s" % args.baseline_model))
#     if os.path.isfile(path_to_prune):
#         mask = np.load(path_to_prune, allow_pickle=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
#     model = models_with_reduce_all_conv.__dict__[args.arch](mask)
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
            args.batch_size = int(args.batch_size / ngpus_per_node)
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

    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

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

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)



if __name__ == '__main__':
    main()