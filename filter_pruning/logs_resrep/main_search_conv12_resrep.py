# Check the performance of the model trained by RESREP.

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

import models_with_reduce_conv12
from ptflops import get_model_complexity_info

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DataDir', default='/home/SSD/Dataset_ImageNet',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True,
                    help='use pre-trained model')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='.', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def main():
    args = parser.parse_args()

    args.path_to_save = args.resume
    
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
    
    args.gpu = gpu

    logger = set_logger(logger_name="{}/_log.txt".format(args.path_to_save))    
    logger.info("This file contains the log of testing the performance of the model trained by RESREP.")
    logger.info("args = %s", args)
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}' with random initial weights".format(args.arch))
        model = models.__dict__[args.arch]()
        
    if args.resume:
        path_to_checkpoint = '{}/sres50_train/latest.pth'.format(args.resume)
        if os.path.isfile(path_to_checkpoint):
            print("=> loading resrep final model")
            checkpoint = torch.load(path_to_checkpoint)
            model = transform_res50(model, checkpoint['model']) # copy weights from resrep
            logger.info("=> loaded resrep final model")
        else:
            raise ValueError("The model to load does not exist!")
    else:
        raise ValueError("The path to resrep trained model does not exist!")

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

    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    num_prunable_params, num_total_params = compute_pruning_upper_bound(model)
    logger.info("prunable/total params (ratio): {:.2f}M/{:.2f}M ({:.2f}%)".format(num_prunable_params/1e6, num_total_params/1e6, num_prunable_params/num_total_params*100))

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    
    val1, val5 = validate(val_loader, model, criterion, logger, args)
    logger.info("validation accuracy of unpruned model: top1 {:.2f}%, top5 {:.2f}%.".format(val1, val5))

    
    for pruning_threshold in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        print_nonzeros_filters(model, logger, pruning_threshold=pruning_threshold)
        model_compressed = compute_and_save_mask(model, args.path_to_save, logger, pruning_threshold=pruning_threshold, arch=args.arch)
        
        val1, val5 = validate(val_loader, model, criterion, logger, args)
        logger.info("validation accuracy of pruned model: top1 {:.2f}%, top5 {:.2f}%.".format(val1, val5))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            torch.save({'small_model': model_compressed.state_dict()}, "{}/small_model_conv12_{}.pth.tar".format(args.path_to_save, pruning_threshold))


def validate(val_loader, model, criterion, logger, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if i % args.print_freq == 0:
                logger.info('valid %04d, loss %.3e, top1 %.2f, top5 %.2f', i, losses.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg

def transform_res50(resnet18, model2):
    key_replace_dict = {
        'layer1.': 'stage1.block',
        'layer2.': 'stage2.block',
        'layer3.': 'stage3.block',
        'layer4.': 'stage4.block',

        'conv1.weight': 'conv1.conv.weight',
        'bn1.': 'conv1.bn.',
        'conv2.weight': 'conv2.conv.weight',
        'bn2.': 'conv2.bn.',
        'conv3.weight': 'conv3.conv.weight',
        'bn3.': 'conv3.bn.',

        '0.downsample.0.weight': 'projection.conv.weight',
        '0.downsample.1.': 'projection.bn.'
    }

    exact_replace_dict = {
        'conv1.weight': 'conv1.conv.weight',
        'bn1.weight': 'conv1.bn.weight',
        'bn1.bias': 'conv1.bn.bias',
        'bn1.running_mean': 'conv1.bn.running_mean',
        'bn1.running_var': 'conv1.bn.running_var'
    }

    def replace_keyword(origin_name):
        for a, b in key_replace_dict.items():
            if a in origin_name:
                return origin_name.replace(a, b)
        return origin_name

    save_dict = {}
    for k, v in resnet18.state_dict().items():
        if k in exact_replace_dict:
            save_dict[k] = model2[exact_replace_dict[k]]
        elif 'downsample' in k:
            save_dict[k] = model2[k.replace('layer', 'stage')
                .replace('0.downsample.0.weight', 'projection.conv.weight')
                .replace('0.downsample.1.', 'projection.bn.')]
        else:
            save_dict[k] = model2[replace_keyword(replace_keyword(replace_keyword(k)))]
            
    resnet18.load_state_dict(save_dict)
    return resnet18
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


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
            
            if "conv1" in name or "conv2" in name:
                num_zero_filters_per_layer[layer - 1]  +=  num_zero_filters_tmp
                num_total_filters_per_layer[layer - 1] += num_total_filters_tmp
            
            num_zero_params += param.numel() * num_zero_filters_tmp / num_total_filters_tmp
        elif "layer" in name and "bn" in name:
            num_zero_params += param.numel() * num_zero_filters_tmp / num_total_filters_tmp
    
    for layer in range(0, 4):
        logger.info("pruning threshold: {}, layer {}: zero/total filters (conv1/2) {}/{} ({}%)".format(pruning_threshold, layer, num_zero_filters_per_layer[layer], num_total_filters_per_layer[layer], num_zero_filters_per_layer[layer]/num_total_filters_per_layer[layer]*100))
        
    logger.info("pruning threshold: {},   total: zero/total filters (conv1/2) {}/{} ({:.2f}%)".format(pruning_threshold, np.sum(num_zero_filters_per_layer), np.sum(num_total_filters_per_layer), np.sum(num_zero_filters_per_layer)/np.sum(num_total_filters_per_layer)*100))
    logger.info("pruning threshold: {}, zero/total filters (ratio): {}/{} ({:.2f}%)".format(pruning_threshold, num_zero_filters, num_total_filters, num_zero_filters/num_total_filters*100))
    logger.info("pruning threshold: {},  zero/total params (ratio): {}/{}M ({:.2f}%)".format(pruning_threshold, num_zero_params/1e6, num_total_params/1e6, num_zero_params/num_total_params*100))
    

def compute_pruning_upper_bound(model):
    num_zero_filters = 0
    num_total_filters = 0    
    
    num_prunable_params = 0
    num_total_params = 0
    
    for name, param in model.named_parameters():
        num_total_params += param.numel()
        if "layer" in name:
            if "conv1" in name or "conv2" in name:
                num_prunable_params += param.numel()

    return num_prunable_params, num_total_params


def compute_and_save_mask(model, file_path, logger, pruning_threshold=1e-6, arch='resnet50'):
    # valid for resnet50 only
    mask = [[[[],[],[]], [[],[],[]], [[],[],[]]], 
            [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]],
            [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]],
            [[[],[],[]], [[],[],[]], [[],[],[]]]]
    for name, param in model.named_parameters():
        if "layer" in name:
            layer_index = int(name[12])
            block_index = int(name[14])
            if "conv" in name:
                conv_index = int(name[20])
                filter_norms = torch.norm(param.view(param.shape[0], -1), p=2, dim=1)
                mask[layer_index-1][block_index][conv_index-1] = (filter_norms.gt(pruning_threshold)).cpu().detach().numpy()

    np.save("{}/mask_{}".format(file_path, pruning_threshold), mask)

    model_compressed = models_with_reduce_conv12.__dict__[arch](mask)
    macs, params = get_model_complexity_info(model_compressed, (3, 224, 224), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    logger.info("pruning threshold: {}, computational complexity: {}, number of parameters: {}".format(pruning_threshold, macs, params))

    model_compressed = transfer_model_parameters(model_compressed, mask, arch, model)

    return model_compressed


def transfer_model_parameters(small_model, mask, arch, big_model):
    if arch == "resnet50":
        indices = [[[[],[],[]], [[],[],[]], [[],[],[]]], 
                  [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]],
                  [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]],
                  [[[],[],[]], [[],[],[]], [[],[],[]]]]
        for layer_index, mask_layer in enumerate(mask):
            for block_index, mask_block in enumerate(mask_layer):
                for conv_index, mask_conv in enumerate(mask_block):
                    for filter_index, mask_filter in enumerate(mask_conv):
                        if mask_filter:
                            indices[layer_index][block_index][conv_index].append(filter_index)
        
    big_state_dict = big_model.state_dict()
    small_state_dict = small_model.state_dict()
    for key, value in small_state_dict.items():
        key_module = "module." + key
        if "layer" in key: #layer1.0.downsample.0.weight
            layer_index = int(key[5]) - 1
            block_index = int(key[7])
            if "conv" in key or "bn" in key:
                if "num_batches_tracked" in key:
                    small_state_dict[key] = big_state_dict[key_module]
                    pass
                else:
                    if "conv" in key:
                        conv_index = int(key[13]) - 1
                    compressed_parameters = big_state_dict[key_module]
                    if conv_index == 0 or conv_index == 1:
                         compressed_parameters = torch.index_select(compressed_parameters, 0, torch.tensor(indices[layer_index][block_index][conv_index]).cuda())
                    if "conv2" in key or "conv3" in key: # if conv_index == 2 or conv_index == 3:
                        compressed_parameters = torch.index_select(compressed_parameters, 1, torch.tensor(indices[layer_index][block_index][conv_index-1]).cuda())
                    small_state_dict[key] = compressed_parameters
            else:
                small_state_dict[key] = big_state_dict[key_module]
        else:
            small_state_dict[key] = big_state_dict[key_module]
    small_model.load_state_dict(small_state_dict)
    
    return small_model

    
if __name__ == '__main__':
    main()