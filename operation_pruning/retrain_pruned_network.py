"""
The operations to be pruned will be removed in the model, so the resulting new model is smaller.
"""

import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import utils_gsparsity
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import random
import json

from genotypes import PRIMITIVES
from model_eval_multipath import NetworkCIFAR as Network_alpha
from model import NetworkCIFAR as Network


CIFAR_CLASSES = 10

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

def load_model(args):
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.cells, args.auxiliary, genotype)
    model = model.cuda()

    model.load_state_dict(torch.load(args.path_to_model))
    return model


def main(args):
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    if args.model_to_resume is None:
        utils.create_exp_dir(args.save)
        RUN_ID = "lr_{}_momentum_{}_mu_{}_time_{}".format(args.learning_rate,
                                                          args.momentum,
                                                          args.weight_decay,
                                                          time.strftime("%Y%m%d-%H%M%S"))
        args.save = "{}/{}-{}".format(args.save, args.save, RUN_ID)
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

        run_data = {}
        run_data['RUN_ID'] = RUN_ID
        run_data['save'] = args.save
        run_data['batch_size'] = args.batch_size
        run_data['pruning_threshold'] = args.pruning_threshold
        run_data['learning_rate'] = args.learning_rate
        run_data['drop_path_prob'] = args.drop_path_prob
        run_data['path_to_model'] = args.path_to_model
        run_data['arch'] = args.arch
        if args.seed is None:
            args.seed = random.randint(0, 10000)
        run_data['seed'] = args.seed

        with open('{}/run_info.json'.format(args.save), 'w') as f:
            json.dump(run_data, f)
    else:
        f = open("{}/run_info.json".format(args.model_to_resume))
        run_data = json.load(f)

        RUN_ID = run_data['RUN_ID']
        args.save = run_data['save']
        args.batch_size = run_data['batch_size']
        args.pruning_threshold = run_data['pruning_threshold']
        args.learning_rate = run_data['learning_rate']
        args.drop_path_prob = run_data['drop_path_prob']
        args.seed = run_data['seed']
        args.path_to_model = run_data['path_to_model']
        args.arch = run_data['arch']

    logger = utils.set_logger(logger_name="{}/_log_{}.txt".format(args.save, RUN_ID))

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logger.info('CARME Slurm ID: {}'.format(os.environ['SLURM_JOBID']))
    logger.info('gpu device = %d' % args.gpu)
    logger.info("args = %s", args)
  
    logger.info("The baseline model is at {}\n".format(args.path_to_model))
        
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    network_eval = network_params(args.init_channels, args.cells, 4, PRIMITIVES, criterion)
    genotype = eval("genotypes.%s" % args.arch)

    args.path_to_model += "/full_weights"
    model_to_prune = load_model(args)
    alpha_network, genotype_network = utils_gsparsity.discretize_model_by_operation(model_to_prune, network_eval, genotype, args.pruning_threshold, args.save)
    logger.info("alpha_network:\n {}".format(alpha_network))
    logger.info("genotype_network:\n {}".format(genotype_network))

    model = Network_alpha(args.init_channels, CIFAR_CLASSES, args.cells, args.auxiliary, genotype_network, alpha_network, network_eval.reduce_cell_indices, network_eval.steps)
    model = model.cuda()

    logger.info("The number of trainable parameters before and after operation pruning is {}M and {}M".format(utils.count_parameters_in_MB(model_to_prune), utils.count_parameters_in_MB(model)))
  
    optimizer = torch.optim.SGD(model.parameters(),
                                args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              float(args.epochs),
                                                              eta_min=args.learning_rate_min)

    if args.model_to_resume is not None:
        checkpoint = torch.load("{}/checkpoint.pth.tar".format(args.model_to_resume))
        model.load_state_dict(checkpoint['latest_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        best_acc = checkpoint['best_acc']
        last_epoch = checkpoint['last_epoch']
        assert last_epoch>=0 and args.epochs>=0 and last_epoch<=args.epochs
    else:
        best_acc = 0
        last_epoch = 0
    
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    if args.model_to_resume is None:
        train_top1 = np.array([])
        train_loss = np.array([])
        valid_top1 = np.array([])
        valid_loss = np.array([])
    else:
        train_top1 = np.load("{}/train_top1.npy".format(args.model_to_resume), allow_pickle=True)
        train_loss = np.load("{}/train_loss.npy".format(args.model_to_resume), allow_pickle=True)
        valid_top1 = np.load("{}/valid_top1.npy".format(args.model_to_resume), allow_pickle=True)
        valid_loss = np.load("{}/valid_loss.npy".format(args.model_to_resume), allow_pickle=True)

    for epoch in range(last_epoch+1, args.epochs+1):
        logger.info('(JOBID %s) epoch %d begins...', os.environ['SLURM_JOBID'], epoch)
        
        model.drop_path_prob = args.drop_path_prob * (epoch-1) / args.epochs
        train_top1_tmp, train_loss_tmp = train(train_queue, model, criterion, optimizer, args, logger)
        valid_top1_tmp, valid_loss_tmp = infer(valid_queue, model, criterion, args, logger)

        train_top1 = np.append(train_top1, train_top1_tmp.item())
        train_loss = np.append(train_loss, train_loss_tmp.item())
        valid_top1 = np.append(valid_top1, valid_top1_tmp.item())
        valid_loss = np.append(valid_loss, valid_loss_tmp.item())

        np.save(args.save+"/train_top1", train_top1)
        np.save(args.save+"/train_loss", train_loss)
        np.save(args.save+"/valid_top1", valid_top1)
        np.save(args.save+"/valid_loss", valid_loss)

        utils.acc_n_loss(train_loss, valid_top1, "{}/acc_n_loss_{}.png".format(args.save, RUN_ID), train_top1, valid_loss)

        is_best = False
        if valid_top1_tmp >= best_acc:
            best_acc = valid_top1_tmp
            is_best = True

        logger.info('(JOBID %s) epoch %d lr %e: train_acc %f, valid_acc %f (best_acc %f)',
                     os.environ['SLURM_JOBID'],
                     epoch,
                     lr_scheduler.get_lr()[0],
                     train_top1_tmp,
                     valid_top1_tmp,
                     best_acc)

        lr_scheduler.step()

        utils.save_checkpoint({'last_epoch': epoch,
                               'latest_model': model.state_dict(),
                               'best_acc': best_acc,
                               'optimizer' : optimizer.state_dict(),
                               'lr_scheduler': lr_scheduler.state_dict()},
                              is_best,
                              args.save)
        
    torch.save(model.state_dict(), "{}/full_weights".format(args.save))        
    logger.info("args = %s", args)

def train(train_queue, model, criterion, optimizer, args, logger):
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
            logger.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, args, logger):
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
                logger.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser("cifar")

    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.035, help='init learning rate')
    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--pruning_threshold', type=float, default=1e-6, help='operation pruning threshold')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--model_to_resume', type=str, default=None, help='path to the model to resume training')
    parser.add_argument('--path_to_model', type=str, help='path to the model to be pruned')
    parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
    
    parser.add_argument('--learning_rate_min', type=float, default=0, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--save', type=str, default='log-retraining', help='experiment name')
    parser.add_argument('--data', type=str, default='/home/SSD/data', help='location of the data corpus')
    parser.add_argument('--report_freq', type=float, default=0, help='report frequency (set 0 to turn off)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--cells', type=int, default=20, help='total number of cells')
    parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

    args = parser.parse_args()

    main(args)