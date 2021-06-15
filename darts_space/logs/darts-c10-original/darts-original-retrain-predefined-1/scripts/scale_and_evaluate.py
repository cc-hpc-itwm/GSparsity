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
from model_multi_path import NetworkCIFAR as Network
from genotypes import PRIMITIVES,Genotype

from ProxSGD_for_weights import ProxSGD
import utils_sparsenas

parser = argparse.ArgumentParser("cifar")

parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
# parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--pruning_threshold', type=float, default=1e-5, help='operation pruning threshold')
parser.add_argument('--use_sgd', action='store_true', default=True, help='new initialization after pruning')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--save', type=str, default='scaling', help='experiment name')

parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

args = parser.parse_args()

utils.create_exp_dir(args.save)
RUN_ID = "lr_{}_momentum_{}_weight_decay_{}_time_{}".format(args.learning_rate,
                                                            args.momentum,
                                                            args.weight_decay,
                                                            time.strftime("%Y%m%d-%H%M%S"))
args.save = "{}/{}-{}".format(args.save, args.save, RUN_ID)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log_{}.txt'.format(RUN_ID)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10    

def get_genotype_from_alpha(alphas, _steps=4, _multiplier=4): # what is _multiplier and why is it 4?
    def _parse():
      gene = []
      n = 2
      start = 0
      for i in range(_steps):
        edges = range(i+2)
        for j in edges:
          k_best = None
          for k in range(len(PRIMITIVES)):
            gene.append((PRIMITIVES[k], j))

      return gene

    all_genes_parsed = _parse()
    all_genes = []
    cellstruc = alphas[0].flatten()
    indices = np.where(cellstruc == 1)[0]
    all_genes.append([all_genes_parsed[x] for x in indices.astype(int)])

    cellstruc = alphas[1].flatten()
    indices = np.where(cellstruc == 1)[0]
    all_genes.append([all_genes_parsed[x] for x in indices.astype(int)])

    gene_normal = all_genes[0]
    gene_reduce = all_genes[1]

    concat = range(2+_steps-_multiplier, _steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype
#####################################################

def main(filename):
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
  
#   alphas = (torch.randint(0,2,(14,7)),torch.randint(0,2,(14,7))) #These are the 2 alphas in the order (normal,reduce)
  alphas = utils_sparsenas.discretize_search_model(filename, threshold=args.pruning_threshold)
  logging.info("Model to discretize is in: {}".format(filename))  
  logging.info("alphas: {}".format(alphas))
  all_genotype = get_genotype_from_alpha(alphas)
    
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, all_genotype, alphas)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

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

#   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  train_acc_trajectory = []
  valid_acc_trajectory = []
  train_obj_trajectory = []
  valid_obj_trajectory = []

  model.drop_path_prob = 0
  valid_acc, valid_obj = infer(valid_queue, model, criterion, args)
  logging.info('(JOBID %s) epoch %d lr %e: train_acc          , valid_acc %f', os.environ['SLURM_JOBID'], 0, scheduler.get_lr()[0], valid_acc)
  valid_acc_trajectory.append(valid_acc)
  valid_obj_trajectory.append(valid_obj)

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

    utils_sparsenas.acc_n_loss(train_obj_trajectory, valid_acc_trajectory, "{}/acc_n_loss_{}.png".format(args.save, RUN_ID), train_acc_trajectory)
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

#     if step % args.report_freq == 0:
#       logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

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

#       if step % args.report_freq == 0:
#         logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_edec_0_rho_0.8_rdec_0_mu_1_time_20201217-093441/full_weights"
  main(model_to_discretize) 