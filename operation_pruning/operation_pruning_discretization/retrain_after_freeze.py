"""
In retrain_after_freeze.py, the operations to be pruned are still preserved in the model, but their weights are frozen to be 0.

In retrain_after_reduce.py, the operations to be pruned are really removed in the model, so the resulting new model is smaller.
"""

import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network

CIFAR_CLASSES = 10

def create_model(args):
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.cells, args.auxiliary, genotype)
    model = model.cuda()
    return model

def load_model(file_pretrained_model, args):
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.cells, args.auxiliary, genotype)
    model = model.cuda()

    model.load_state_dict(torch.load(file_pretrained_model))
    return model

def freeze_new_model(model, model_pretrained, threshold=1e-6): # for new initialization
    num_param_before = 0
    num_param_after = 0
    indices_zero_operations = []
    num_zero_operations = 0
    
    for param, param_pretrained in zip(model.parameters(), model_pretrained.parameters()):
        if param_pretrained.requires_grad:
            num_param_before += param_pretrained.numel()
            if torch.norm(param_pretrained) <= threshold:                
                param.data = torch.zeros_like(param)
                param.requires_grad = False
            else:
                num_param_after += param.numel()
    
    return model, num_param_before, num_param_after

def freeze_old_model(model, threshold=1e-6): # with no new initialization
    num_param_before = 0
    num_param_after = 0
    for param in model.parameters():
        if param.requires_grad:
            num_param_before += param.numel()
            if torch.norm(param) <= threshold:                
                param.data = torch.zeros_like(param.data)
                param.requires_grad = False
            else:
                num_param_after += param.numel()
    return model, num_param_before, num_param_after


def main(args, file_pretrained_model, model_to_resume=None):
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  print("args = %s", args)
  
  print("The baseline model is at {}\n".format(file_pretrained_model))
  model = load_model(file_pretrained_model, args)
  model, num_param_before, num_param_after = freeze_old_model(model, threshold=args.pruning_threshold)
    
  print("The number of trainable parameters before and after operation pruning is {} MB and {} MB".format(num_param_before/1e6, num_param_after/1e6))
        
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  model.drop_path_prob = 0
  valid_acc, valid_obj = infer(valid_queue, model, criterion)
  print("accuracy after pruning: {}".format(valid_acc))


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input).cuda()
      target = Variable(target).cuda()

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data, n)
      top1.update(prec1.data, n)
      top5.update(prec5.data, n)

  return top1.avg, objs.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser("cifar")

    parser.add_argument('--data', type=str, default='/home/yangy/data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--report_freq', type=float, default=0, help='report frequency (set 0 to turn off)')
    parser.add_argument('--plot_freq', type=int, default=1, help='report frequency (set 0 to turn off)')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--cells', type=int, default=20, help='total number of cells')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

    parser.add_argument('--pruning_threshold', type=float, default=1e-3, help='operation pruning threshold')

    args = parser.parse_args()

#     file_pretrained_model = '../darts-pruning/darts-pruning-lr_0.001_edec_0_rho_0.9_rdec_0_mu_0.0001_time_20210202-145002/full_weights'
    
#     file_pretrained_model = '../darts-pruning/darts-pruning-lr_0.001_edec_0_rho_0.9_rdec_0_mu_0.0002_time_20210208-163708/full_weights'

#     file_pretrained_model = '../darts-pruning/darts-pruning-lr_0.001_edec_0_rho_0.9_rdec_0_mu_0.0005_time_20210202-145112/full_weights'
    
    file_pretrained_model = '../darts-pruning/darts-pruning-lr_0.001_edec_0_rho_0.9_rdec_0_mu_0.002_time_20210202-145342/full_weights'
    
#     file_pretrained_model = '../darts-pruning/darts-pruning-lr_0.001_edec_0_rho_0.9_rdec_0_mu_0.004_time_20210202-162931/full_weights'

    main(args, file_pretrained_model) 