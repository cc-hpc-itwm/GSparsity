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
from model_search import Network
from architect import Architect
import utils_sparsenas

from ProxSGD_for_cell_search import ProxSGD
from ProxSGD_for_operations import ProxSGD as ProxSGD_for_operations

def main(args):
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  utils.create_exp_dir(args.save)
  RUN_ID = "lr_{}_edec_{}_rho_{}_rdec_{}_mu_{}_time_{}".format(args.learning_rate, args.epsilon_decay, args.rho, args.rho_decay, args.weight_decay, time.strftime("%Y%m%d-%H%M%S"))
  args.save = "{}/{}-{}".format(args.save, args.save, RUN_ID)
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, '_log_{}.txt'.format(RUN_ID)))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  CIFAR_CLASSES = 10
    
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('CARME Slurm ID: {}'.format(os.environ['SLURM_JOBID']))
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    
  if args.search_prune:
    optimizer = ProxSGD_for_operations(model.parameters(), 
                                       lr=args.learning_rate, 
                                       epsilon_decay=args.epsilon_decay, 
                                       rho_decay=args.rho_decay, 
                                       weight_decay=args.weight_decay, 
                                       gamma=args.gamma, 
                                       clip_bounds=(0,1),
                                       betas=(args.rho,0.999))
#     optimizer = torch.optim.SGD(model.parameters(),
#                                 args.learning_rate,
#                                 momentum=args.rho,
#                                 weight_decay=args.weight_decay)  
  else: # search_for_stage or search_for_cell
    if args.search_for_stage:
      model_params = utils_sparsenas.group_model_params_by_stage(model, mu=args.weight_decay, reduce_cell_indices=[2, 5], num_edges=14, num_ops=7)
    elif args.search_for_cell:
      model_params = utils_sparsenas.group_model_params_by_cell(model, mu=args.weight_decay, reduce_cell_indices=[2, 5], num_edges=14, num_ops=7)
    else:
      raise ValueError("None of search_for_stage, search_for_cell and search_prune is selected.")  
    optimizer = ProxSGD(model_params, 
                        lr=args.learning_rate, 
                        epsilon_decay=args.epsilon_decay, 
                        rho_decay=args.rho_decay, 
                        weight_decay=args.weight_decay, 
                        gamma=args.gamma, 
                        clip_bounds=(0,1),
                        betas=(args.rho,0.999), 
                        normalization=args.normalization)    

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

#   train_transform, valid_transform = utils._data_transforms_cifar10(args)
#   train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

#   num_train = len(train_data)
#   indices = list(range(num_train))
#   split = int(np.floor(args.train_portion * num_train))

#   train_queue = torch.utils.data.DataLoader(
#       train_data, batch_size=args.batch_size,
#       sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
#       pin_memory=True, num_workers=4)

#   valid_queue = torch.utils.data.DataLoader(
#       train_data, batch_size=args.batch_size,
#       sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
#       pin_memory=True, num_workers=4)

  #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
  #      optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  #architect = Architect(model, args)

  utils_sparsenas.plot_individual_op_norm(model, args.save+"/operator_norm_individual_{}_epoch_{:03d}.png".format(RUN_ID, 0), figsize_width=200)

  if args.search_for_stage:
    utils_sparsenas.plot_op_norm_across_stages(model_params, args.save+"/operator_norm_stage_{}_epoch_{:03d}.png".format(RUN_ID, 0))
  if args.search_for_cell:
    utils_sparsenas.plot_op_norm_across_cells(model_params, args.save+"/operator_norm_cell_{}_epoch_{:03d}".format(RUN_ID, 0))

  train_acc_trajectory = []
  valid_acc_trajectory = []
  train_obj_trajectory = []
  valid_obj_trajectory = []
    
  for epoch in range(1, args.epochs + 1):
    train_acc, train_obj = train(train_queue, valid_queue, model, criterion, optimizer, args.learning_rate, args.report_freq)
    valid_acc, valid_obj = infer(valid_queue, model, criterion, args.report_freq)

    logging.info('(JOBID %s) epoch %d lr %e: train_acc %f, valid_acc %f', 
                 os.environ['SLURM_JOBID'], 
                 epoch, 
                 args.learning_rate, 
                 train_acc, 
                 valid_acc)
    
    train_acc_trajectory.append(train_acc)
    train_obj_trajectory.append(train_obj)
    valid_acc_trajectory.append(valid_acc)
    valid_obj_trajectory.append(valid_obj)
    
    utils_sparsenas.acc_n_loss(train_obj_trajectory, valid_acc_trajectory, "{}/acc_n_loss_{}.png".format(args.save, RUN_ID), train_acc_trajectory)
    np.save(args.save+"/train_accuracies",train_acc_trajectory)
    np.save(args.save+"/train_objvals", train_obj_trajectory)
    np.save(args.save+"/test_accuracies",valid_acc_trajectory)
    np.save(args.save+"/test_objvals", valid_obj_trajectory)
    torch.save(model.state_dict(), "{}/full_weights".format(args.save))
  
    if epoch % 1 == 0: #Plot Group sparsity after 10 epochs
      utils_sparsenas.plot_individual_op_norm(model, args.save+"/individual_operator_norm_{}_epoch_{:03d}.png".format(RUN_ID, epoch), figsize_width=200)

      if args.search_for_stage:
          utils_sparsenas.plot_op_norm_across_stages(model_params, args.save+"/operator_norm_stage_{}_epoch_{:03d}.png".format(RUN_ID, epoch))
      if args.search_for_cell:
          utils_sparsenas.plot_op_norm_across_cells(model_params, args.save+"/operator_norm_cell_{}_epoch_{:03d}".format(RUN_ID, epoch))

  logging.info("args = %s", args)


def train(train_queue, valid_queue, model, criterion, optimizer, lr, report_freq):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
      
        n = input.size(0)

        input = input.cuda()
        target = target.cuda()#async=True)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

#         if step % report_freq == 0:
#           logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

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
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

#             if step % report_freq == 0:
#                 logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    # search for a single cell structure (normal cell and reduction cell)
    parser = argparse.ArgumentParser("cifar")
    
    parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--rho', type=float, default=0.8, help='init rho')
    parser.add_argument('--weight_decay', type=float, default=0.3, help='weight decay (mu)')
    parser.add_argument('--normalization', default=False, help='True for normalized regularization (mu)')
    parser.add_argument('--search_for_stage', action='store_true', default=False, help='Search for Stage if True')
    parser.add_argument('--search_for_cell', action='store_true', default=True, help='Search for Cell if True')
    parser.add_argument('--search_prune', action='store_true', default=False, help='Prune operations of the train-search model if True')
    parser.add_argument('--save', type=str, default='darts-search-pruning', help='experiment name')
    parser.add_argument('--remark', type=str, default="no remark", help='remark about the experiment')

    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    parser.add_argument('--epsilon_decay', type=float, default=0, help='init epsilon decay')
    parser.add_argument('--rho_decay', type=float, default=0, help='init rho decay')
    parser.add_argument('--gamma', type=int, default=0, help='init gamma')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
#     parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
#     parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
#     parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
#     parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
#     parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    args = parser.parse_args()
    
    assert(sum([args.search_for_stage, args.search_for_cell, args.search_prune])) == 1, "One and only one of the following flags can be true: search_for_stage, search_for_cell and search_prune."
    
    main(args)   