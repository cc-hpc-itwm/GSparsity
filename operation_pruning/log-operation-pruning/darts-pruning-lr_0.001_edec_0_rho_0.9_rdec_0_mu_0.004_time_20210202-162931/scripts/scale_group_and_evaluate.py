import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from genotypes import PRIMITIVES

import utils
import utils_sparsenas
from model_eval_multipath import NetworkCIFAR as Network
from ProxSGD_for_weights import ProxSGD

class network_params():
    def __init__(self, init_channels, cells, steps, operations, criterion):
        self.init_channels = init_channels
        self.cells = cells
        self.steps = steps #the number of nodes between input nodes and the output node
        self.num_edges = sum([i+2 for i in range(steps)]) #14
        self.ops = operations
        self.num_ops = len(operations)
        self.reduce_cell_indices = [cells//3, (2*cells)//3]
        self.criterion = criterion

def main(args, model_to_discretize, model_to_resume=None):
  utils.create_exp_dir(args.save)
  RUN_ID = "lr_{}_momentum_{}_weight_decay_{}_cells_{}_time_{}".format(args.learning_rate,
                                                                       args.momentum,
                                                                       args.weight_decay,
                                                                       args.cells,
                                                                       time.strftime("%Y%m%d-%H%M%S"))
  args.save = "{}/{}-{}".format(args.save, args.save, RUN_ID)
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, '_log_{}.txt'.format(RUN_ID)))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  CIFAR_CLASSES = 10

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
  
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
    
  network_search = network_params(args.init_channels_search, args.cells_search, 4, PRIMITIVES, criterion)
  network_eval = network_params(args.init_channels, args.cells, 4, PRIMITIVES, criterion)
  if args.scale_type == "cell":
      alpha_network, genotype_network = utils_sparsenas.discretize_search_model_by_cell(model_to_discretize, network_eval, network_search, threshold=args.pruning_threshold) #alpha for each cell
  elif args.scale_type == "stage":
      alpha_network, genotype_network = utils_sparsenas.discretize_search_model_by_stage(model_to_discretize, network_eval, network_search, threshold=args.pruning_threshold, swap_stage=args.swap_stage)
        
  assert len(alpha_network) == args.cells, "Each cell should have its individual alpha."
  logging.info("Model to discretize is in: {}\n".format(model_to_discretize))
  logging.info("alpha_network:\n {}".format(alpha_network))
  logging.info("genotype_network:\n {}".format(genotype_network))
    
  utils_sparsenas.visualize_cell_in_network(network_eval, alpha_network, args.scale_type, args.save)

  model = Network(args.init_channels, CIFAR_CLASSES, args.cells, args.auxiliary, genotype_network, alpha_network)
  model = model.cuda()    

#   testimage = torch.randn(1,3,32,32).cuda() #batchsize, channel, weight*height
#   model.drop_path_prob = 0
#   outTest, _ = model(testimage)
#   time.sleep(3600)

  assert args.last_epoch>=0 and args.epochs>=0 and args.last_epoch<=args.epochs
  if args.last_epoch > 0:
    if model_to_resume is None:
        raise ValueError("The model to resume training is not provided!")
    else:
        logging.info("model to resume training is: {}".format(model_to_resume))
        model.load_state_dict(torch.load(model_to_resume))
    
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

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
  if args.last_epoch > 0: # adapt the learning rate to the beginning epoch
    for i in range(1, args.last_epoch+1):
        logging.info('dummy epoch %d lr %e', i, scheduler.get_lr()[0])
        scheduler.step()

  train_acc_trajectory = []
  train_obj_trajectory = []
  valid_acc_trajectory = []
  valid_obj_trajectory = []
  for epoch in range(args.last_epoch+1, args.epochs+1):
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
    
#     time.sleep(3600)
    
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
    
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--remark', type=str, default="no remark", help='remark about the experiment')
    parser.add_argument('--save', type=str, default=None, help='experiment name')
    parser.add_argument('--last_epoch', type=int, default=0, help='the beginning epoch (new training if 0, or resume an existing training if >0)')
    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--pruning_threshold', type=float, default=1e-2, help='operation pruning threshold')
    parser.add_argument('--cells', type=int, default=14, help='total number of cells')
    parser.add_argument('--scale_type', choices=["cell", "stage"], 
                        default="cell", help='scale type: scale cell, scale stage')
    parser.add_argument('--swap_stage', action='store_true', default=False, help='swap stage when scaling (to test the importance of the ordering of stages found in search)')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    
    parser.add_argument('--learning_rate_min', type=float, default=0, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--use_sgd', action='store_true', default=True, help='new initialization after pruning')
    parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--report_freq', type=float, default=0, help='report frequency (set 0 to turn off)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--init_channels_search', type=int, default=16, help='num of init channels')
    parser.add_argument('--cells_search', type=int, default=8, help='total number of cells')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    if args.save is None:
        if args.scale_type == "stage":
            if args.swap_stage:
                args.save = "scaling-swapped-stage"
            else:
                args.save = "scaling-stage"
        elif args.scale_type == "cell":
            args.save = "scaling-cell"

    if args.scale_type == "stage":
#         model_to_discretize = "search-for-stage/search-for-stage-lr_0.001_rho_0.8_mu_800_div_1_GC_0_time_20210110-195648/full_weights"
#         model_to_discretize = "search-for-stage/search-for-stage-lr_0.001_rho_0.8_mu_900_div_1_GC_0_time_20210115-210753/full_weights"
        model_to_discretize = "search-for-stage/search-for-stage-lr_0.001_rho_0.8_mu_800_div_1_GC_0_time_20210121-102011/full_weights"
#         model_to_discretize = "search-for-stage/search-for-stage-lr_0.001_rho_0.8_mu_800_div_1_GC_0_time_20210121-102454/full_weights"
    elif args.scale_type == "cell":
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_10000_div_1_GC_0_time_20210117-203741/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_11000_div_1_GC_0_time_20210117-212817/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_12000_div_1_GC_0_time_20210117-212828/full_weights"
        model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_12000_div_1_GC_0_time_20210120-111931/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_12000_div_1_GC_0_time_20210125-103025/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_12000_div_1_GC_0_time_20210125-103214/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_40000_div_1_GC_0_time_20210120-112055/full_weights"

    if args.last_epoch > 0:
        model_to_resume = "scaling-cell/scaling-cell-lr_0.025_momentum_0.9_weight_decay_0.0003_cells_14_time_20210125-164501/full_weights"
        main(args, model_to_discretize, model_to_resume) 
    else:
        main(args, model_to_discretize)