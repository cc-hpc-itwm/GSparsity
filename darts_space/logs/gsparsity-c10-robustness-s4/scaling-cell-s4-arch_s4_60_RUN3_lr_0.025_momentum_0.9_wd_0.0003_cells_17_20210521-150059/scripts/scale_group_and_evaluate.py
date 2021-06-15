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
import random
import json

from genotypes import spaces_dict
import list_of_models as list_of_models
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
        self.num_ops = len(operations['primitives_normal'][0])
        self.reduce_cell_indices = [cells//3, (2*cells)//3]
        self.criterion = criterion

def main(args):
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    f_search = open("{}/run_info.json".format(eval("list_of_models.%s" % args.arch)))
    run_data_search = json.load(f_search)
    search_space = run_data_search['search_space']
        
    if args.model_to_resume is None:
        if search_space is None:
            pass
        else:
            args.save += "-{}".format(search_space)
        utils.create_exp_dir(args.save)
        RUN_ID = "arch_{}_lr_{}_momentum_{}_wd_{}_cells_{}_{}".format(args.arch,
                                                                      args.learning_rate,
                                                                      args.momentum,
                                                                      args.weight_decay,
                                                                      args.cells,
                                                                      time.strftime("%Y%m%d-%H%M%S"))
        args.save = "{}/{}-{}".format(args.save, args.save, RUN_ID)
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

        run_data = {}
        run_data['RUN_ID'] = RUN_ID
        run_data['save'] = args.save
        run_data['batch_size'] = args.batch_size
        run_data['pruning_threshold'] = args.pruning_threshold
        run_data['cells'] = args.cells
        run_data['learning_rate'] = args.learning_rate
        run_data['scale_type'] = args.scale_type
        run_data['swap_stage'] = args.swap_stage
        run_data['drop_path_prob'] = args.drop_path_prob
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
        args.cells = run_data['cells']
        args.learning_rate = run_data['learning_rate']
        args.scale_type = run_data['scale_type']
        args.swap_stage = run_data['swap_stage']
        args.drop_path_prob = run_data['drop_path_prob']
        args.arch = run_data['arch']
        args.seed = run_data['seed']

    if search_space is None:
        PRIMITIVES = spaces_dict['darts']        
    else:
        PRIMITIVES = spaces_dict[search_space]
        
        
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

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    
    network_search = network_params(args.init_channels_search,
                                    args.cells_search,
                                    4,
                                    PRIMITIVES,
                                    criterion)
    network_eval = network_params(args.init_channels,
                                  args.cells,
                                  4,
                                  PRIMITIVES,
                                  criterion)
    
    model_to_discretize = "{}/model_final".format(eval("list_of_models.%s" % args.arch))
    if args.scale_type == "cell":
        alpha_network, genotype_network = utils_sparsenas.discretize_search_model_by_cell(model_to_discretize,
                                                                                          network_eval,
                                                                                          network_search,
                                                                                          threshold=args.pruning_threshold) #alpha for each cell
    elif args.scale_type == "stage":
        alpha_network, genotype_network = utils_sparsenas.discretize_search_model_by_stage(model_to_discretize,
                                                                                           network_eval,
                                                                                           network_search,
                                                                                           threshold=args.pruning_threshold,
                                                                                           swap_stage=args.swap_stage)
        
    assert len(alpha_network) == args.cells, "Each cell should have its individual alpha."
    logger.info("Model to discretize is in: {}\n".format(model_to_discretize))
    logger.info("alpha_network:\n {}".format(alpha_network))
    logger.info("genotype_network:\n {}".format(genotype_network))

    utils_sparsenas.visualize_cell_in_network(network_eval, alpha_network, args.scale_type, args.save)

    model = Network(args.init_channels,
                    CIFAR_CLASSES,
                    args.cells,
                    args.auxiliary,
                    genotype_network,
                    alpha_network,
                    network_eval.reduce_cell_indices,
                    network_eval.steps,
                    PRIMITIVES
                   )
    model = model.cuda()

    logger.info("param size = %fM", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(model.parameters(),
                                args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              float(args.epochs),
                                                              eta_min=args.learning_rate_min)
    
    if args.model_to_resume is not None:
        train_top1 = np.load("{}/train_top1.npy".format(args.model_to_resume), allow_pickle=True)
        train_loss = np.load("{}/train_loss.npy".format(args.model_to_resume), allow_pickle=True)
        valid_top1 = np.load("{}/valid_top1.npy".format(args.model_to_resume), allow_pickle=True)
        valid_loss = np.load("{}/valid_loss.npy".format(args.model_to_resume), allow_pickle=True)
        
        checkpoint = torch.load("{}/checkpoint.pth.tar".format(args.model_to_resume))
        model.load_state_dict(checkpoint['latest_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        best_top1 = checkpoint['best_top1']
        last_epoch = checkpoint['last_epoch']
        assert last_epoch>=0 and args.epochs>=0 and last_epoch<=args.epochs
    else:
        train_top1 = np.array([])
        train_loss = np.array([])
        valid_top1 = np.array([])
        valid_loss = np.array([])
        
        best_top1 = 0
        last_epoch = 0
    
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(train_data,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=2)
    valid_queue = torch.utils.data.DataLoader(valid_data,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)

    
    for epoch in range(last_epoch+1, args.epochs+1):
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

        utils_sparsenas.acc_n_loss(train_loss,
                                   valid_top1,
                                   "{}/acc_n_loss_{}.png".format(args.save, RUN_ID),
                                   train_top1,
                                   valid_loss)

        is_best = False
        if valid_top1_tmp >= best_top1:
            best_top1 = valid_top1_tmp
            is_best = True

        logger.info('(JOBID %s) epoch %d lr %e: train_top1 %f, valid_top1 %f (best_top1 %f)',
                     os.environ['SLURM_JOBID'],
                     epoch,
                     lr_scheduler.get_lr()[0],
                     train_top1_tmp,
                     valid_top1_tmp,
                     best_top1)

        lr_scheduler.step()

        utils.save_checkpoint({'last_epoch': epoch,
                               'latest_model': model.state_dict(),
                               'best_top1': best_top1,
                               'optimizer' : optimizer.state_dict(),
                               'lr_scheduler': lr_scheduler.state_dict()},
                              is_best,
                              args.save)
    
    torch.save(model.state_dict(), "{}/model_final".format(args.save))
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
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--save', type=str, default=None, help='experiment name')
    parser.add_argument('--batch_size', type=int, default=80, help='batch size')
    parser.add_argument('--pruning_threshold', type=float, default=1e-5, help='operation pruning threshold')
    parser.add_argument('--cells', type=int, default=14, help='total number of cells')
    parser.add_argument('--scale_type', choices=["cell", "stage"], 
                        default="cell", help='scale type: scale cell, scale stage')
    parser.add_argument('--swap_stage', action='store_true', default=False, help='swap stage when scaling (to test the importance of the ordering of stages found in search)')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--arch', type=str, help='which arch to discretize and scale')
    parser.add_argument('--model_to_resume', type=str, default=None, help='path to the model to resume training')
    
    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    parser.add_argument('--learning_rate_min', type=float, default=0, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
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
    parser.add_argument('--seed', type=int, default=None, help='random integer seed between 0 and 10000 if None')
    args = parser.parse_args()

    if args.save is None:
        if args.scale_type == "stage":
            if args.swap_stage:
                args.save = "scaling-swapped-stage"
            else:
                args.save = "scaling-stage"
        elif args.scale_type == "cell":
            args.save = "scaling-cell"

    main(args)