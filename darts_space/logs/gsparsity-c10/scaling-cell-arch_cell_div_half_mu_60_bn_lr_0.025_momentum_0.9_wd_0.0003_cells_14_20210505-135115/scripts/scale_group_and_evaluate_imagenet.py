import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import json
from collections import OrderedDict

from torch.autograd import Variable

from genotypes import PRIMITIVES
import list_of_models as list_of_models
import utils
import utils_sparsenas
from model_eval_multipath import NetworkImageNet as Network

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

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main(args):
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)
    
    if args.model_to_resume is None:
        utils.create_exp_dir(args.save)
        RUN_ID = "arch_{}_lr_{}_momentum_{}_wd_{}_cells_{}_{}".format(args.arch,
                                                                      args.learning_rate,
                                                                      args.momentum,
                                                                      args.weight_decay,
                                                                      args.cells,
                                                                      time.strftime("%Y%m%d-%H%M%S")
                                                                     )
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
        args.arch = run_data['arch']
        args.seed = run_data['seed']

    logger = utils.set_logger(logger_name="{}/_log_{}.txt".format(args.save, RUN_ID))

    IMAGENET_CLASSES = 1000
    
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
    criterion_smooth = CrossEntropyLabelSmooth(IMAGENET_CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    network_search = network_params(args.init_channels_search,
                                    args.cells_search,
                                    4,
                                    PRIMITIVES,
                                    criterion)
    network_eval = network_params(args.init_channels,
                                  args.cells,
                                  4,
                                  PRIMITIVES,
                                  criterion_smooth)

    model_to_discretize = "{}/full_weights".format(eval("list_of_models.%s" % args.arch))
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

    utils_sparsenas.visualize_cell_in_network(network_eval,
                                              alpha_network,
                                              args.scale_type,
                                              args.save)

    model = Network(args.init_channels,
                    IMAGENET_CLASSES,
                    args.cells,
                    args.auxiliary,
                    genotype_network,
                    alpha_network,
                    network_eval.reduce_cell_indices,
                    network_eval.steps)
    model = model.cuda()
    
    optimizer = torch.optim.SGD(model.parameters(),
                                args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   args.decay_period,
                                                   gamma=args.gamma)

    if args.model_to_resume is not None:
        model.load_state_dict(torch.load("{}/full_weights".format(args.model_to_resume)))
        checkpoint = torch.load("{}/checkpoint.pth.tar".format(args.model_to_resume))
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        best_acc_top1 = checkpoint['best_acc_top1']
        model.drop_path_prob = checkpoint['drop_path_prob']
        last_epoch = checkpoint['last_epoch']
        assert last_epoch>=0 and args.epochs>=0 and last_epoch<=args.epochs
    else:
        last_epoch = 0
        best_acc_top1 = 0
        model.drop_path_prob = 0
        
    if args.parallel:
        model = nn.DataParallel(model)
        
    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(traindir,
                                  transforms.Compose([
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ColorJitter(
                                          brightness=0.4,
                                          contrast=0.4,
                                          saturation=0.4,
                                          hue=0.2),
                                      transforms.ToTensor(),
                                      normalize]))
    valid_data = dset.ImageFolder(validdir,
                                  transforms.Compose([
                                      transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize]))

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    if args.model_to_resume is None:
        train_acc_trajectory = np.array([])
        train_obj_trajectory = np.array([])
        valid_top1_trajectory = np.array([])
        valid_top5_trajectory = np.array([])
        valid_obj_trajectory = np.array([])
    else:
        train_acc_trajectory = np.load("{}/train_accuracies.npy".format(args.model_to_resume), allow_pickle=True)
        train_obj_trajectory = np.load("{}/train_objvals.npy".format(args.model_to_resume), allow_pickle=True)
        valid_top1_trajectory = np.load("{}/test_top1_accuracies.npy".format(args.model_to_resume), allow_pickle=True)
        valid_top5_trajectory = np.load("{}/test_top5_accuracies.npy".format(args.model_to_resume), allow_pickle=True)
        valid_obj_trajectory = np.load("{}/test_objvals.npy".format(args.model_to_resume), allow_pickle=True)

    for epoch in range(last_epoch+1, args.epochs+1):
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer, logger)

        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion, logger)

        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
            if args.parallel:
                torch.save(model.module.state_dict(), "{}/best_model".format(args.save))
            else:
                torch.save(model.state_dict(), "{}/best_model".format(args.save))
        else:
            is_best = False

        train_acc_trajectory = np.append(train_acc_trajectory, train_acc.item())
        train_obj_trajectory = np.append(train_obj_trajectory, train_obj.item())
        valid_top1_trajectory = np.append(valid_top1_trajectory, valid_acc_top1.item())
        valid_top5_trajectory = np.append(valid_top5_trajectory, valid_acc_top5.item())
        valid_obj_trajectory = np.append(valid_obj_trajectory, valid_obj.item())

        np.save(args.save+"/train_accuracies", train_acc_trajectory)
        np.save(args.save+"/train_objvals", train_obj_trajectory)
        np.save(args.save+"/test_top1_accuracies", valid_top1_trajectory)
        np.save(args.save+"/test_top5_accuracies", valid_top5_trajectory)
        np.save(args.save+"/test_objvals", valid_obj_trajectory)

        utils_sparsenas.acc_n_loss(train_obj_trajectory,
                                   valid_top1_trajectory,
                                   "{}/acc_n_loss_{}.png".format(args.save, RUN_ID),
                                   train_acc_trajectory,
                                   valid_obj_trajectory)

        if args.parallel:
            torch.save(model.module.state_dict(), "{}/full_weights".format(args.save))
        else:
            torch.save(model.state_dict(), "{}/full_weights".format(args.save))

        logger.info('(JOBID %s) epoch %d lr %e: train_acc %f, valid_acc_top1 %f (best_acc_top1 %f), valid_acc_top5 %f',
                     os.environ['SLURM_JOBID'], 
                     epoch, 
                     lr_scheduler.get_lr()[0], 
                     train_acc, 
                     valid_acc_top1,
                     best_acc_top1,
                     valid_acc_top5)

        lr_scheduler.step()    

        utils.save_checkpoint({'last_epoch': epoch,
                               'best_acc_top1': best_acc_top1,
                               'optimizer' : optimizer.state_dict(),
                               'lr_scheduler': lr_scheduler.state_dict(),
                               'drop_path_prob': model.drop_path_prob},
                              False,
                              args.save)

    logger.info("args = %s", args)

def train(train_queue, model, criterion, optimizer, logger):
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

        if step % args.report_freq == 0:
            logger.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, logger):
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

            if step % args.report_freq == 0:
                logger.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser("imagenet")
    parser.add_argument('--cells', type=int, default=8, help='total number of cells')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--parallel', action='store_true', default=True, help='data parallelism')
    parser.add_argument('--arch', type=str, help='which architecture to choose from list_of_models.py')
    parser.add_argument('--pruning_threshold', type=float, default=1e-1, help='operation pruning threshold')
    parser.add_argument('--model_to_resume', type=str, default=None, help='path to the model to resume training')
    
    parser.add_argument('--scale_type', choices=["cell", "stage"], 
                        default="cell", help='scale type: scale cell, scale stage')
    parser.add_argument('--swap_stage', action='store_true', default=False, help='swap stage when scaling (to test the importance of the ordering of stages found in search)')
    parser.add_argument('--data', type=str, default='/home/yangy/Dataset_ImageNet/', help='location of the data corpus')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=1000, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
    parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
    parser.add_argument('--save', type=str, default=None, help='experiment name')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--init_channels_search', type=int, default=16, help='num of init channels')
    parser.add_argument('--cells_search', type=int, default=8, help='total number of cells')
    args = parser.parse_args()
    
    if args.save is None:
        if args.scale_type == "stage":
            if args.swap_stage:
                args.save = "scaling-swapped-stage-imagenet"
            else:
                args.save = "scaling-stage-imagenet"
        elif args.scale_type == "cell":
            args.save = "scaling-cell-imagenet"
            
    main(args) 