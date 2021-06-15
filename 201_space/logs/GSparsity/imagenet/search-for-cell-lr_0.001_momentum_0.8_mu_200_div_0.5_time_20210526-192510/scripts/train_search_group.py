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
#from model_search_multipath import Network
from generic_model import GenericNAS201Model as Network
from extras import get_cell_based_tiny_net, get_search_spaces 
from config_utils import dict2config
import utils_sparsenas
#from genotypes import PRIMITIVES
from cell_operations import NAS_BENCH_201 as PRIMITIVES
import random
import json

from ProxSGD_for_groups import ProxSGD
from ProxSGD_for_operations import ProxSGD as ProxSGD_for_operations
from get_dataset_with_transform import get_datasets

class network_params():
    def __init__(self, init_channels, cells, steps, operations):
        self.init_channels = init_channels
        self.cells = cells
        self.steps = steps #the number of nodes between input nodes and the output node
        self.num_edges = sum([i+2 for i in range(steps)]) #14
        self.num_ops = len(operations)
        self.reduce_cell_indices = [cells//3, (2*cells)//3]

def main(args):
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    if args.model_to_resume is None:
        utils.create_exp_dir(args.save)
        RUN_ID = "lr_{}_momentum_{}_mu_{}_{}_{}_time_{}".format(args.learning_rate,
                                                                args.momentum,
                                                                args.weight_decay,
                                                                args.normalization,
                                                                args.normalization_exponent,
                                                                time.strftime("%Y%m%d-%H%M%S"))
        args.save = "{}/{}-{}".format(args.save, args.save, RUN_ID)
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
        
        run_data = {}
        run_data['RUN_ID'] = RUN_ID
        run_data['save'] = args.save
        run_data['batch_size'] = args.batch_size
        run_data['learning_rate'] = args.learning_rate
        run_data['learning_rate_min'] = args.learning_rate_min
        run_data['momentum'] = args.momentum
        run_data['weight_decay'] = args.weight_decay
        run_data['normalization'] = args.normalization
        run_data['normalization_exponent'] = args.normalization_exponent
        run_data['use_lr_scheduler'] = args.use_lr_scheduler
        run_data['search_type'] = args.search_type
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
        args.learning_rate = run_data['learning_rate']
        args.learning_rate_min = run_data['learning_rate_min']
        args.momentum = run_data['momentum']
        args.weight_decay = run_data['weight_decay']
        args.normalization = run_data['normalization']
        args.normalization_exponent = run_data['normalization_exponent']
        args.use_lr_scheduler = run_data['use_lr_scheduler']
        args.search_type = run_data['search_type']
        args.seed = run_data['seed']

    logger = utils.set_logger(logger_name="{}/_log_{}.txt".format(args.save, RUN_ID))

    CIFAR_CLASSES = 120
    
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

    """NeW"""
    search_space = get_search_spaces("tss", "nats-bench")

    model_config = dict2config(
        dict(
            name="generic",
            C=16,
            N=5,
            max_nodes=4,
            num_classes=120,
            space=search_space,
            affine=bool(0),
            track_running_stats=bool(0),
        ),
        None,
    )
    model = get_cell_based_tiny_net(model_config)
    model = model.cuda()
    #pathio = "search-for-cell/search-for-cell-lr_0.001_momentum_0.8_mu_0.1_mul_0.5_time_20210523-152749/model_final"
    #model.load_state_dict(torch.load(pathio))
    #i=0
    #for cell in model._cells:
    #    print(i)
    #    i+=1
    #print(search_model)
    
    #######################
    #model = Network(args.init_channels, CIFAR_CLASSES, args.cells, criterion)
    #model = model.cuda()
    #print(model)
    #print(jjjj)

    if args.search_type is None:
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=3e-4)
    elif args.search_type == "operation":
        optimizer = ProxSGD_for_operations(model.parameters(),
                                           lr=args.learning_rate,
                                           momentum=args.momentum,
                                           weight_decay=args.weight_decay,
                                           normalization=args.normalization,
                                           normalization_exponent=args.normalization_exponent
                                          )
    else: # search_for_stage or search_for_cell
        print(PRIMITIVES)
        network_search = network_params(args.init_channels,
                                        args.cells,
                                        4,
                                        PRIMITIVES)
        if args.search_type == "stage":
            model_params = utils_sparsenas.group_model_params_by_stage(model,
                                                                       network_search,
                                                                       mu=args.weight_decay)
        elif args.search_type == "cell":
            model_params = utils_sparsenas.group_model_params_by_cell(model,
                                                                      network_search,
                                                                      mu=args.weight_decay)
        optimizer = ProxSGD(model_params,
                            lr=args.learning_rate, 
                            weight_decay=args.weight_decay, 
                            clip_bounds=(0, 1),
                            momentum=args.momentum, 
                            normalization=args.normalization,
                            normalization_exponent=args.normalization_exponent)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              float(args.epochs),
                                                              eta_min=args.learning_rate_min)
    
    train_data, valid_data, xshape, class_num = get_datasets("ImageNet16-120", args.data, -1)
    #train_data, valid_data, xshape, class_num = get_datasets("cifar100", args.data, -1)
    
    #print(xshape,class_num)
    #train_transform, valid_transform = utils._data_transforms_cifar10(args)
    #train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    if args.train_portion == 1:
        #valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
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
    else:
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))
        train_queue = torch.utils.data.DataLoader(train_data,
                                                  batch_size=args.batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                                  pin_memory=True,
                                                  num_workers=4)
        valid_queue = torch.utils.data.DataLoader(train_data,
                                                  batch_size=args.batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
                                                  pin_memory=True,
                                                  num_workers=4)
    
    if args.model_to_resume is not None:
        train_top1 = np.load("{}/train_top1.npy".format(args.model_to_resume), allow_pickle=True)
        train_loss = np.load("{}/train_loss.npy".format(args.model_to_resume), allow_pickle=True)
        valid_top1 = np.load("{}/valid_top1.npy".format(args.model_to_resume), allow_pickle=True)
        valid_loss = np.load("{}/valid_loss.npy".format(args.model_to_resume), allow_pickle=True)        
        if args.search_type == "cell" or "stage":
            train_regl = np.load("{}/train_regl.npy".format(args.model_to_resume), allow_pickle=True)
        
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
        if args.search_type == "cell" or "stage":
            train_regl = np.array([])
        
        best_top1 = 0
        last_epoch = 0
        
        if args.plot_freq > 0:
            utils_sparsenas.plot_individual_op_norm(model, "{}/operator_norm_individual_{}_epoch_{:03d}.png".format(args.save, RUN_ID, 0))
            if args.search_type == "stage":
                utils_sparsenas.plot_op_norm_across_stages(model_params, "{}/operator_norm_stage_{}_epoch_{:03d}.png".format(args.save, RUN_ID, 0))
            if args.search_type == "cell":
                utils_sparsenas.plot_op_norm_across_cells(model_params, "{}/operator_norm_cell_{}_epoch_{:03d}".format(args.save, RUN_ID, 0))

    #utils_sparsenas.plot_op_norm_across_cells(model_params, "testitwithout")
    #utils_sparsenas.plot_op_norm_across_cells(model_params, "testitdiv",normalization="div",normalization_exponent=0.5)        
    #utils_sparsenas.plot_op_norm_across_cells(model_params, "testitmul",normalization="mul",normalization_exponent=0.5)
    #print(jjj)
    logger.info("param size = %fM", utils.count_parameters_in_MB(model))

    for epoch in range(last_epoch+1, args.epochs + 1):
        #network.set_drop_path(float(epoch) / args.epochs, xargs.drop_path_rate)
        logger.info('(JOBID %s) epoch %d begins...', os.environ['SLURM_JOBID'], epoch)
        train_top1_tmp, train_loss_tmp = train(train_queue, valid_queue, model, criterion, optimizer, args.learning_rate, args.report_freq, logger)
        valid_top1_tmp, valid_loss_tmp = infer(valid_queue, model, criterion, args.report_freq, logger)

        train_top1 = np.append(train_top1, train_top1_tmp.item())
        train_loss = np.append(train_loss, train_loss_tmp.item())
        valid_top1 = np.append(valid_top1, valid_top1_tmp.item())
        valid_loss = np.append(valid_loss, valid_loss_tmp.item())

        np.save(args.save+"/train_top1", train_top1)
        np.save(args.save+"/train_loss", train_loss)
        np.save(args.save+"/valid_top1", valid_top1)
        np.save(args.save+"/valid_loss", valid_loss)

        is_best = False
        if valid_top1_tmp >= best_top1:
            best_top1 = valid_top1_tmp
            is_best = True

        logger.info('(JOBID %s) epoch %d lr %.3e: train_top1 %.2f, valid_top1 %.2f (best_top1 %.2f)',
                    os.environ['SLURM_JOBID'],
                    epoch,
                    lr_scheduler.get_lr()[0],
                    train_top1_tmp,
                    valid_top1_tmp,
                    best_top1)

        if args.search_type == "cell":
            train_regl_tmp = utils_sparsenas.get_regularization_term(model_params, args)
            train_regl = np.append(train_regl, train_regl_tmp.item())
            np.save(args.save+"/train_regl", train_regl)

            logger.info('(JOBID %s) epoch %d obj_val %.2f: loss %.2f + regl %.2f (%.2f * %.2f)',
                        os.environ['SLURM_JOBID'],
                        epoch,
                        train_loss_tmp + args.weight_decay * train_regl_tmp,
                        train_loss_tmp,
                        args.weight_decay * train_regl_tmp,
                        args.weight_decay,
                        train_regl_tmp)
            utils_sparsenas.acc_n_loss(train_loss,
                                       valid_top1,
                                       "{}/acc_n_loss_{}.png".format(args.save, RUN_ID),
                                       train_top1,
                                       valid_loss,
                                       train_loss + args.weight_decay * train_regl
                                      )
        else:
            utils_sparsenas.acc_n_loss(train_loss,
                                       valid_top1,
                                       "{}/acc_n_loss_{}.png".format(args.save, RUN_ID),
                                       train_top1,
                                       valid_loss
                                      )

        
        if args.use_lr_scheduler:
            lr_scheduler.step()

        utils.save_checkpoint({'last_epoch': epoch,
                               'latest_model': model.state_dict(),
                               'best_top1': best_top1,
                               'optimizer' : optimizer.state_dict(),
                               'lr_scheduler': lr_scheduler.state_dict()},
                              is_best,
                              args.save)

        if args.plot_freq > 0 and epoch % args.plot_freq == 0: #Plot group sparsity after args.plot_freq epochs
            '''comment plot_individual_op_norm to save time'''
            if args.search_type == "operation":
                utils_sparsenas.plot_individual_op_norm(model,
                                                        "{}/operator_norm_individual_{}_epoch_{:03d}.png".format(args.save, RUN_ID, epoch))
            if args.search_type == "stage":
                utils_sparsenas.plot_op_norm_across_stages(model_params,
                                                           "{}/operator_norm_stage_{}_epoch_{:03d}.png".format(args.save, RUN_ID, epoch))
            if args.search_type == "cell":
                utils_sparsenas.plot_op_norm_across_cells(model_params,
                                                          "{}/operator_norm_cell_{}_epoch_{:03d}".format(args.save, RUN_ID, epoch))

        torch.save(model.state_dict(), "{}/model_final".format(args.save))
        
    if args.search_type == "cell" or "stage":
        utils_sparsenas.plot_individual_op_norm(model,
                                                "{}/operator_norm_individual_{}_epoch_{:03d}.png".format(args.save, RUN_ID, epoch))
    logger.info("args = %s", args)


def train(train_queue, valid_queue, model, criterion, optimizer, lr, report_freq, logger):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    
    for step, (input, target) in enumerate(train_queue):      
        input = input.cuda()
        target = target.cuda()#async=True)

        optimizer.zero_grad()
        #print(input)
        _,logits = model(input)
        #print(logits)
        loss = criterion(logits, target)
        print(loss)
        loss.backward()        
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)            
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
        
        if report_freq > 0 and step % report_freq == 0:
            logger.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg
  
def infer(valid_queue, model, criterion, report_freq, logger):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()#async=True)

            _,logits = model(input)
            loss = criterion(logits, target)
            

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if report_freq > 0:
                if step % report_freq == 0:
                    logger.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    # search for a single cell structure (normal cell and reduction cell)
    parser = argparse.ArgumentParser("cifar")
    
    parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.8, help='init momentum')
    parser.add_argument('--weight_decay', type=float, default=200, help='weight decay (mu)')
    parser.add_argument('--search_type', choices=[None, "cell", "stage", "operation"], 
                        default="cell", help='search type: search for cell, search for stage, search for operation (prune operations)')
    parser.add_argument('--normalization', choices=["none", "mul", "div"], 
                        default="div", help='normalize the regularization (mu) by operation dimension: none, mul or div')
    parser.add_argument('--normalization_exponent', type=float, 
                        default=0.5, help='normalization exponent to normalize the weight decay (mu)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--use_lr_scheduler', action='store_true', default=False, help='use lr scheduler')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--model_to_resume', type=str, default=None, help='path to the pretrained model to prune')
    
    parser.add_argument('--save', type=str, default=None, help='experiment name (default None)')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--train_portion', type=float, default=1, help='portion of training data (if 1, the validation set is the test set)')
    parser.add_argument('--grad_clip', type=float, default=0, help='gradient clipping (set 0 to turn off)')    
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--cells', type=int, default=8, help='total number of cells')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--report_freq', type=int, default=0, help='report frequency (set 0 to turn off)')
    parser.add_argument('--plot_freq', type=int, default=1, help='plot (operation norm) frequency (set 0 to turn off)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
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
    
    main(args)