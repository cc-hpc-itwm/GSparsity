"""
In retrain_after_freeze.py, the operations to be pruned are still preserved in the model, but their weights are frozen to be 0.

In retrain_after_reduce.py, the operations to be pruned are already removed in the model, so the resulting new model is smaller.
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

from ProxSGD_for_weights import ProxSGD
import utils_sparsenas

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


def main(args, file_pretrained_model):
  utils.create_exp_dir(args.save)
  RUN_ID = "lr_{}_momentum_{}_weight_decay_{}_new_init_{}_time_{}".format(args.learning_rate,
                                                                          args.momentum,
                                                                          args.weight_decay,
                                                                          args.new_initialization,
                                                                          time.strftime("%Y%m%d-%H%M%S"))
  args.save = "{}/{}-{}".format(args.save, args.save, RUN_ID)
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, '_log_{}.txt'.format(RUN_ID)))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('CARME Slurm ID: {}'.format(os.environ['SLURM_JOBID']))
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  
  logging.info("The baseline model is at {}\n".format(file_pretrained_model))
  if args.new_initialization:
    model = create_model(args)
    model_pretrained = load_model(file_pretrained_model, args)
    model, num_param_before, num_param_after = freeze_new_model(model, model_pretrained, threshold=args.pruning_threshold)
    logging.info("NEW initialization after operation pruning")
  else:
    model = load_model(file_pretrained_model, args)
    model, num_param_before, num_param_after = freeze_old_model(model, threshold=args.pruning_threshold)
    logging.info("NO new initialization after operation pruning")
    
  logging.info("The number of trainable parameters before and after operation pruning is {} MB and {} MB".format(num_param_before/1e6, num_param_after/1e6))
#   logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  if args.use_sgd:
      optimizer = torch.optim.SGD(
          model.parameters(),
          args.learning_rate,
          momentum=args.momentum,
          weight_decay=args.weight_decay
          )
  else:
    optimizer = ProxSGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, mu=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  utils_sparsenas.plot_individual_op_norm(model, args.save+"/individual_operator_norm_{}_epoch_{:03d}.png".format(RUN_ID, 0))
  train_acc_trajectory, train_obj_trajectory = [], []
  valid_acc_trajectory, valid_obj_trajectory = [], []
  for epoch in range(1, args.epochs+1):
    model.drop_path_prob = args.drop_path_prob * (epoch-1) / args.epochs
    train_acc, train_obj = train(train_queue, model, criterion, optimizer, args)
    valid_acc, valid_obj = infer(valid_queue, model, criterion, args)
    
    logging.info('(JOBID %s) epoch %d lr %e: train_acc %f, valid_acc %f', 
                 os.environ['SLURM_JOBID'], 
                 epoch, 
                 scheduler.get_lr()[0], 
                 train_acc, 
                 valid_acc)

    scheduler.step()
    
    train_acc_trajectory.append(train_acc)
    train_obj_trajectory.append(train_obj)
    valid_acc_trajectory.append(valid_acc)
    valid_obj_trajectory.append(valid_obj)
    
    np.save(args.save+"/train_accuracies",train_acc_trajectory)
    np.save(args.save+"/train_objvals", train_obj_trajectory)
    np.save(args.save+"/test_accuracies",valid_acc_trajectory)
    np.save(args.save+"/test_objvals", valid_obj_trajectory)

    utils_sparsenas.acc_n_loss(train_obj_trajectory, valid_acc_trajectory, "{}/acc_n_loss_{}.png".format(args.save, RUN_ID), train_acc_trajectory, valid_obj_trajectory)
    torch.save(model.state_dict(), "{}/full_weights".format(args.save))
    
    if args.plot_freq > 0 and epoch % args.plot_freq == 0: #Plot Group sparsity every args.plot_freq epochs
      utils_sparsenas.plot_individual_op_norm(model, args.save+"/individual_operator_norm_{}_epoch_{:03d}.png".format(RUN_ID, epoch))      
    
  logging.info("args = %s", args)

def train(train_queue, model, criterion, optimizer, args):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    if args.report_freq > 0 and step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, args):
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

    if args.report_freq > 0 and step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser("cifar")

    parser.add_argument('--learning_rate_min', type=float, default=0, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--save', type=str, default='darts-freeze-retraining', help='experiment name')
    parser.add_argument('--use_sgd', action='store_true', default=True, help='new initialization after pruning')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--report_freq', type=float, default=0, help='report frequency (set 0 to turn off)')
    parser.add_argument('--plot_freq', type=int, default=1, help='report frequency (set 0 to turn off)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--cells', type=int, default=20, help='total number of cells')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

    parser.add_argument('--learning_rate', type=float, default=0.035, help='init learning rate')
    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--pruning_threshold', type=float, default=1e-6, help='operation pruning threshold')
    parser.add_argument('--new_initialization', action='store_true', default=True, help='new initialization after pruning')

    args = parser.parse_args()

    file_pretrained_model = "darts-pruning/darts-pruning-lr_0.001_edec_0_rho_0.9_rdec_0_mu_0.0005_time_20210202-145112/full_weights"
#     file_pretrained_model = "darts-pruning/darts-pruning-lr_0.001_edec_0_rho_0.9_rdec_0_mu_0.005_time_20210106-134801/full_weights"
#     file_pretrained_model = "darts-pruning/darts-pruning-lr_0.001_edec_0_rho_0.9_rdec_0_mu_0.001_time_20210129-211739/full_weights"
    main(args, file_pretrained_model) 