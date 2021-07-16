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
import json
import random

from torch.autograd import Variable
from model import NetworkCIFAR as Network

from ProxSGD_for_operations import ProxSGD

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
        args.save = "{}/{}-network_{}-{}".format(args.save, args.save, args.arch, RUN_ID)
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

        run_data = {}
        run_data['RUN_ID'] = RUN_ID
        run_data['save'] = args.save
        run_data['learning_rate'] = args.learning_rate
        run_data['learning_rate_min'] = args.learning_rate_min
        run_data['use_lr_scheduler'] = args.use_lr_scheduler
        run_data['momentum'] = args.momentum
        run_data['weight_decay'] = args.weight_decay
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
        args.learning_rate = run_data['learning_rate']
        args.learning_rate_min = run_data['learning_rate_min']
        args.use_lr_scheduler = run_data['use_lr_scheduler']
        args.momentum = run_data['momentum']
        args.weight_decay = run_data['weight_decay']
        args.seed = run_data['seed']
        args.arch = run_data['arch']

    logger = utils.set_logger(logger_name="{}/_log_{}.txt".format(args.save, RUN_ID))
    
    CIFAR_CLASSES = 10

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logger.info('CARME Slurm ID: {}'.format(os.environ['SLURM_JOBID']))
    logger.info('gpu device = %d' % args.gpu)
    logger.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.cells, args.auxiliary, genotype)
    model = model.cuda()

    logger.info("param size = %fM", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = ProxSGD(model.parameters(),
                        lr=args.learning_rate,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              float(args.epochs),
                                                              eta_min=args.learning_rate_min)

    if args.model_to_resume is not None:
        checkpoint = torch.load("{}/checkpoint.pth.tar".format(args.model_to_resume))
        model.load_state_dict(checkpoint['full_weights'])
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

    utils.plot_individual_op_norm(model, args.save+"/individual_operator_norm_{}_epoch_{:03d}.png".format(RUN_ID, 0))
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
        logger.info('(JOBID %s) epoch %d begins...',
                     os.environ['SLURM_JOBID'], 
                     epoch)

        model.drop_path_prob = args.drop_path_prob * (epoch - 1) / args.epochs

        train_top1_tmp, train_loss_tmp = train(train_queue, model, criterion, optimizer, args, logger)
        valid_top1_tmp, valid_loss_tmp = infer(valid_queue, model, criterion, args, logger)

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

        if args.use_lr_scheduler:
            lr_scheduler.step()

        train_top1 = np.append(train_top1, train_top1_tmp.item())
        train_loss = np.append(train_loss, train_loss_tmp.item())
        valid_top1 = np.append(valid_top1, valid_top1_tmp.item())
        valid_loss = np.append(valid_loss, valid_loss_tmp.item())

        np.save(args.save+"/train_top1", train_top1)
        np.save(args.save+"/train_loss", train_loss)
        np.save(args.save+"/valid_top1", valid_top1)
        np.save(args.save+"/valid_loss", valid_loss)

        utils.acc_n_loss(train_loss, valid_top1, "{}/acc_n_loss_{}.png".format(args.save, RUN_ID), train_top1, valid_loss)

        if args.plot_freq > 0 and epoch % args.plot_freq == 0: #Plot Group sparsity every args.plot_freq epochs
            utils.plot_individual_op_norm(model, args.save+"/individual_operator_norm_{}_epoch_{:03d}.png".format(RUN_ID, epoch))      

        utils.save_checkpoint({'last_epoch': epoch,
                               'full_weights': model.state_dict(),
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
    parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='minimum learning rate')
    parser.add_argument('--use_lr_scheduler', action='store_true', default=True, help='use learning rate scheduler')
    parser.add_argument('--momentum', type=float, default=0.9, help='init momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0004, help='regularization gain mu')
    parser.add_argument('--save', type=str, default='log-operation-pruning', help='experiment name')
    parser.add_argument('--model_to_resume', type=str, default=None, help='path to the model to resume training')

    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')

    parser.add_argument('--data', type=str, default='/home/SSD/data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--report_freq', type=int, default=0, help='report frequency (set 0 to turn off)')
    parser.add_argument('--plot_freq', type=int, default=1, help='report frequency (set 0 to turn off)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--cells', type=int, default=20, help='total number of cells')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--arch', type=str, default='DARTS_V2', help='which architecture to use')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

    args = parser.parse_args()

    main(args) 