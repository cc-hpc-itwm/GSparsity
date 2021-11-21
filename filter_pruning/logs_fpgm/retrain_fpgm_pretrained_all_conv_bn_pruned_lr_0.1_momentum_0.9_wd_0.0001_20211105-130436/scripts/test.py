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
parser.add_argument('--data', metavar='DataDir', default='/home/yangy/Dataset_ImageNet',
                    help='path to dataset')
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
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-p', '--print-freq', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
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
parser.add_argument('--path_to_save', type=str, default="conv12", help='path to the folder where the experiment will be saved')
parser.add_argument('--run_id', type=str, default=None, help='the identifier of this specific run')


best_acc1 = 0


def main():
    args = parser.parse_args()
    resume = "imagenet-retrain_resnet50_mul_mu_0_00006_epoch64_lr_0.1_momentum_0.9_wd_0.0001_20211102-102027"
    if resume:
        f = open("{}/run_info.json".format(resume))
        run_info = json.load(f)
        vars(args).update(run_info['args'])
#         args.data='/home/SSD/Dataset_ImageNet'
        args.path_to_save="retrain_mul_mu_0_00006_epoch64_lr_0.1_momentum_0.9_wd_0.0001_20211102-102027"
        
        run_info = {}
        run_info['args'] = vars(args)        
        with open('{}/run_info.json'.format(args.path_to_save), 'w') as f:
            json.dump(run_info, f)
    
    
if __name__ == '__main__':
    main()