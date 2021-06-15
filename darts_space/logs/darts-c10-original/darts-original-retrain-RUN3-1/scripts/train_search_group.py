# This file searches for a group structure that will be scaled up to form the full network for evaluation (retraining).
# Depending on the singularity of the group, a group may consist of a cell, a stage or an operation:
# search for a cell (the same operation in different cells is in the same group)
# search for a stage (the same operation in different stages is in the same group. An example of the stage is normal_cell normal_cell (stage 1) reduce_cell (stage 2) normal_cell normal_cell(stage 3) reduce_cell (stage 4))
# search for an operation, which is equivalent to pruning operations. As a result, the operations remained in each cell could be different

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
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model_search_multipath import Network
import utils_sparsenas
from genotypes import PRIMITIVES
import random

from ProxSGD_for_groups import ProxSGD
from ProxSGD_for_operations import ProxSGD as ProxSGD_for_operations

class network_params():
    def __init__(self, init_channels, cells, steps, operations):
        self.init_channels = init_channels
        self.cells = cells
        self.steps = steps #the number of nodes between input nodes and the output node
        self.num_edges = sum([i+2 for i in range(steps)]) #14
        self.num_ops = len(operations)
        self.reduce_cell_indices = [cells//3, (2*cells)//3]

def main(args, pretrained_model_path=None):
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  utils.create_exp_dir(args.save)
  RUN_ID = "lr_{}_rho_{}_mu_{}_{}_{}_GC_{}_time_{}".format(args.learning_rate,
                                                                            args.momentum, 
                                                                            args.weight_decay,
                                                                            args.normalization,
                                                                            args.normalization_exponent, 
                                                                            args.grad_clip,
                                                                            time.strftime("%Y%m%d-%H%M%S"))
  args.save = "{}/{}-{}".format(args.save, args.save, RUN_ID)
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, '_log_{}.txt'.format(RUN_ID)))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  CIFAR_CLASSES = 10
    
  random_seed = random.randint(1, 10000)
  np.random.seed(random_seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(random_seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(random_seed)

  logging.info('CARME Slurm ID: {}'.format(os.environ['SLURM_JOBID']))
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  logging.info("random seed is {}".format(random_seed))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.cells, criterion)
  model = model.cuda()

  if pretrained_model_path is not None:
    logging.info('Using pretrained model stored at: {}\n'.format(pretrained_model_path))
    model.load_state_dict(torch.load(pretrained_model_path))
    
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        
  if args.search_type == "operation":
    optimizer = ProxSGD_for_operations(model.parameters(), 
                                       lr=args.learning_rate, 
                                       momentum=args.momentum,
                                       weight_decay=args.weight_decay, 
                                       clip_bounds=(0,1))
#     optimizer = torch.optim.SGD(model.parameters(),
#                                 lr = args.learning_rate,
#                                 momentum = args.momentum,
#                                 weight_decay = args.weight_decay)  
  else: # search_for_stage or search_for_cell
    network_search = network_params(args.init_channels, args.cells, 4, PRIMITIVES)
    if args.search_type == "stage":
      model_params = utils_sparsenas.group_model_params_by_stage(model, network_search, mu=args.weight_decay)
    elif args.search_type == "cell":
      model_params = utils_sparsenas.group_model_params_by_cell(model, network_search, mu=args.weight_decay)
    
    optimizer = ProxSGD(model_params, 
                        lr=args.learning_rate, 
                        weight_decay=args.weight_decay, 
                        clip_bounds=(0,1),
                        momentum=args.momentum, 
                        normalization=args.normalization,
                        normalization_exponent=args.normalization_exponent)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  if args.train_portion == 1:
      valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

      train_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

      valid_queue = torch.utils.data.DataLoader(
          valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
  else:
      num_train = len(train_data)
      indices = list(range(num_train))
      split = int(np.floor(args.train_portion * num_train))

      train_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
          pin_memory=True, num_workers=4)

      valid_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
          pin_memory=True, num_workers=4)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
       optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  utils_sparsenas.plot_individual_op_norm(model, args.save+"/operator_norm_individual_{}_epoch_{:03d}.png".format(RUN_ID, 0))
  if args.search_type == "stage":
    utils_sparsenas.plot_op_norm_across_stages(model_params, args.save+"/operator_norm_stage_{}_epoch_{:03d}.png".format(RUN_ID, 0))
  if args.search_type == "cell":
    utils_sparsenas.plot_op_norm_across_cells(model_params, args.save+"/operator_norm_cell_{}_epoch_{:03d}".format(RUN_ID, 0))

  train_acc_trajectory, train_obj_trajectory = [], []
  valid_acc_trajectory, valid_obj_trajectory = [], []
  for epoch in range(1, args.epochs + 1):
    train_acc, train_obj = train(train_queue, valid_queue, model, criterion, optimizer, args.learning_rate, args.report_freq)
    valid_acc, valid_obj = infer(valid_queue, model, criterion, args.report_freq)

    logging.info('(JOBID %s) epoch %d lr %e: train_acc %f, valid_acc %f', 
                 os.environ['SLURM_JOBID'], 
                 epoch, 
                 scheduler.get_lr()[0],
                 train_acc, 
                 valid_acc)
    
    if args.use_lr_scheduler:
        scheduler.step()

    train_acc_trajectory.append(train_acc)
    train_obj_trajectory.append(train_obj)
    valid_acc_trajectory.append(valid_acc)
    valid_obj_trajectory.append(valid_obj)
    
    utils_sparsenas.acc_n_loss(train_obj_trajectory, valid_acc_trajectory, "{}/acc_n_loss_{}.png".format(args.save, RUN_ID), train_acc_trajectory, valid_obj_trajectory)
    
    np.save(args.save+"/train_accuracies",train_acc_trajectory)
    np.save(args.save+"/train_objvals", train_obj_trajectory)
    np.save(args.save+"/test_accuracies",valid_acc_trajectory)
    np.save(args.save+"/test_objvals", valid_obj_trajectory)
    
    torch.save(model.state_dict(), "{}/full_weights".format(args.save))
  
    if args.plot_freq > 0 and epoch % args.plot_freq == 0: #Plot Group sparsity after 10 epochs
      utils_sparsenas.plot_individual_op_norm(model, args.save+"/operator_norm_individual_{}_epoch_{:03d}.png".format(RUN_ID, epoch))
      if args.search_type == "stage":
          utils_sparsenas.plot_op_norm_across_stages(model_params, args.save+"/operator_norm_stage_{}_epoch_{:03d}.png".format(RUN_ID, epoch))
      if args.search_type == "cell":
          utils_sparsenas.plot_op_norm_across_cells(model_params, args.save+"/operator_norm_cell_{}_epoch_{:03d}".format(RUN_ID, epoch))

  logging.info("args = %s", args)


def train(train_queue, valid_queue, model, criterion, optimizer, lr, report_freq):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    
    for step, (input, target) in enumerate(train_queue):      
        input = input.cuda()
        target = target.cuda()#async=True)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()        
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)            
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
        
        if report_freq > 0:
            if step % report_freq == 0:
              logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg
  
def infer(valid_queue, model, criterion, report_freq):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()#async=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if report_freq > 0:
                if step % report_freq == 0:
                    logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    # search for a single cell structure (normal cell and reduction cell)
    parser = argparse.ArgumentParser("cifar")
    
    parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.8, help='init momentum')

    parser.add_argument('--weight_decay', type=float, default=55, help='weight decay (mu)')
    parser.add_argument('--search_type', choices=["cell", "stage", "operation"], 
                        default="cell", help='search type: search for cell, search for stage, search for operation (prune operations)')
    parser.add_argument('--normalization', choices=["none", "mul", "div"], 
                        default="div", help='normalize the regularization (mu) by operation dimension: none, mul or div')
    parser.add_argument('--normalization_exponent', type=float, 
                        default=0.5, help='normalization exponent to normalize the weight decay (mu)')
    
    parser.add_argument('--save', type=str, default=None, help='experiment name (default None)')
    parser.add_argument('--remark', type=str, default="no remark", help='remark about the experiment')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--use_pretrained_model', action='store_true', default=False, help='Start from a pretrained model if True')
    parser.add_argument('--train_portion', type=float, default=1, help='portion of training data (if 1, the validation set is the test set)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--use_lr_scheduler', action='store_true', default=False, help='use cutout')
    parser.add_argument('--grad_clip', type=float, default=0, help='gradient clipping (set 0 to turn off)')
    
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--cells', type=int, default=8, help='total number of cells')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--report_freq', type=int, default=0, help='report frequency (set 0 to turn off)')
    parser.add_argument('--plot_freq', type=int, default=1, help='plot (operation norm) frequency (set 0 to turn off)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
#     parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    
    args = parser.parse_args()
    
    if args.save is None:
        if args.search_type == "stage":
            args.save = "search-for-stage"
        elif args.search_type == "cell":
            args.save = "search-for-cell"
        elif args.search_type == "operation":
            args.save = "darts-search-pruning"
    
    if args.normalization == "none":
        args.normalization_exponent = 0
    
    if args.use_pretrained_model:
        pretrained_model_path = "darts-search-pruning-with-sgd/darts-search-pruning-with-sgd-lr_0.001_rho_0.9_mu_0.0003_time_20201228-150357/full_weights"
        main(args, pretrained_model_path=pretrained_model_path)
    else:
        main(args)