# This file searches for a cell structure that will be scaled up to form the full network for evaluation (retraining).

import os
import sys
import time
import glob
import random
import json
import argparse
import numpy as np

import torch
import torch.utils
from torch import nn
from torchvision import datasets as dset
from torch.backends import cudnn
from torchvision import transforms

import utils
from model_search_imagenet import Network
import utils_sparsenas_imagenet as utils_sparsenas
from genotypes import spaces_dict
from ProxSGD_for_groups import ProxSGD

class network_params():
    def __init__(self, init_channels, cells, steps, operations):
        self.init_channels = init_channels
        self.cells = cells
        self.steps = steps  # the number of nodes between input nodes and the output node
        self.num_edges = sum([i+2 for i in range(steps)])  # 14
        self.ops = operations
        self.num_ops = len(operations['primitives_normal'][0])
        self.reduce_cell_indices = [cells//3, (2*cells)//3]

def main(args):
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    if args.path_to_resume is None:
        resume_training = False
        run_id = "mu_{}_{}_{}_time_{}".format(args.weight_decay,
                                              args.normalization,
                                              args.normalization_exponent,
                                              time.strftime("%Y%m%d-%H%M%S"))
        args.path_to_save = "{}-space-{}-{}".format(args.path_to_save, args.search_space, run_id)
        args.seed = random.randint(0, 10000) if args.seed is None else args.seed

        run_info = {}
        run_info['run_id'] = run_id
        run_info['args'] = vars(args)

        utils.create_exp_dir(args.path_to_save, scripts_to_save=glob.glob('*.py'))
        with open('{}/run_info.json'.format(args.path_to_save), 'w') as f:
            json.dump(run_info, f)
    else:
        resume_training = True
        f = open("{}/run_info.json".format(args.path_to_resume))
        run_info = json.load(f)

        run_id = run_info['run_id']
        vars(args).update(run_info['args'])

    logger = utils.set_logger(logger_name="{}/_log_{}.txt".format(args.path_to_save, run_id))

    PRIMITIVES = spaces_dict[args.search_space]
    NUM_CLASSES = 1000

    np.random.seed(args.seed)
    #torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)

    logger.info('CARME Slurm ID: {}'.format(os.environ['SLURM_JOBID']))
    logger.info('gpu device = %d' % args.gpu)
    logger.info("args = %s", args)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # dataset split
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    train_data = dset.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_data = dset.ImageFolder(valdir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    num_train = len(train_data)
    num_val = len(valid_data)
    print('# images to train network: %d' % num_train)
    print('# images to validate network: %d' % num_val)

    model = Network(args.init_channels, NUM_CLASSES, args.cells,
                    PRIMITIVES)
    model = nn.DataParallel(model)
    model = model.cuda()

    logger.info("param size = %fM", utils.count_parameters_in_MB(model))

    if args.search_type is None:
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=3e-4)
    else:  # search_for_stage or search_for_cell
        network_search = network_params(args.init_channels,
                                        args.cells,
                                        4,
                                        PRIMITIVES)
        if args.search_type == "cell":
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

    if args.train_portion == 1:
        train_queue = torch.utils.data.DataLoader(train_data,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=args.workers)
        valid_queue = torch.utils.data.DataLoader(valid_data,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=args.workers)
    else:
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))
        train_queue = torch.utils.data.DataLoader(train_data,
                                                  batch_size=args.batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                                  pin_memory=True,
                                                  num_workers=args.workers)
        valid_queue = torch.utils.data.DataLoader(train_data,
                                                  batch_size=args.batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
                                                  pin_memory=True,
                                                  num_workers=args.workers)

    if resume_training:
        train_top1 = np.load("{}/train_top1.npy".format(args.path_to_save), allow_pickle=True)
        train_loss = np.load("{}/train_loss.npy".format(args.path_to_save), allow_pickle=True)
        valid_top1 = np.load("{}/valid_top1.npy".format(args.path_to_save), allow_pickle=True)
        valid_loss = np.load("{}/valid_loss.npy".format(args.path_to_save), allow_pickle=True)
        if args.search_type == "cell":
            train_regl = np.load("{}/train_regl.npy".format(args.path_to_save), allow_pickle=True)

        checkpoint = torch.load("{}/checkpoint.pth.tar".format(args.path_to_save))
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
        if args.search_type == "cell":
            train_regl = np.array([])

        best_top1 = 0
        last_epoch = 0

    lr = args.learning_rate
    for epoch in range(last_epoch+1, args.epochs+1):
        if epoch < 6 and args.batch_size > 32:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * epoch / 5.0
            logger.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch) / 5.0)

        train_top1_tmp, train_loss_tmp = train(train_queue, model, criterion, optimizer, args.learning_rate, args.report_freq, logger)
        train_top1 = np.append(train_top1, train_top1_tmp.item())
        train_loss = np.append(train_loss, train_loss_tmp.item())
        np.save(args.path_to_save+"/train_top1", train_top1)
        np.save(args.path_to_save+"/train_loss", train_loss)

        is_best = False
        if epoch >= args.epochs - 2:
            valid_top1_tmp, valid_loss_tmp = infer(valid_queue, model, criterion, args.report_freq, logger)
            valid_top1 = np.append(valid_top1, valid_top1_tmp.item())
            valid_loss = np.append(valid_loss, valid_loss_tmp.item())
            np.save(args.path_to_save+"/valid_top1", valid_top1)
            np.save(args.path_to_save+"/valid_loss", valid_loss)

            if valid_top1_tmp >= best_top1:
                best_top1 = valid_top1_tmp
                is_best = True

        if args.search_type is None:
            utils_sparsenas.acc_n_loss(train_loss,
                                       valid_top1,
                                       "{}/acc_n_loss_{}.png".format(args.path_to_save, run_id),
                                       train_top1,
                                       valid_loss
                                      )
        elif args.search_type == "cell":
            train_regl_tmp = utils_sparsenas.get_regularization_term(model_params, args)
            train_regl = np.append(train_regl, train_regl_tmp.item())
            np.save(args.path_to_save+"/train_regl", train_regl)

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
                                       "{}/acc_n_loss_{}.png".format(args.path_to_save, run_id),
                                       train_top1,
                                       valid_loss,
                                       train_loss + args.weight_decay * train_regl
                                      )

        current_lr = lr_scheduler.get_lr()[0]
        if args.use_lr_scheduler:
            lr_scheduler.step()

        utils.save_checkpoint({'last_epoch': epoch,
                               'latest_model': model.state_dict(),
                               'best_top1': best_top1,
                               'optimizer': optimizer.state_dict(),
                               'lr_scheduler': lr_scheduler.state_dict()},
                              is_best,
                              args.path_to_save)

        if args.plot_freq > 0 and epoch % args.plot_freq == 0:  # Plot group sparsity after args.plot_freq epochs
            # comment plot_individual_op_norm to save time
            if args.search_type == "cell":
                utils_sparsenas.plot_op_norm_across_cells(model_params,
                                                          "{}/operator_norm_cell_{}_epoch_{:03d}".format(args.path_to_save, run_id, epoch))

        torch.save(model.state_dict(), "{}/model_final".format(args.path_to_save))

        if epoch >= args.epochs - 2:
            logger.info('(JOBID %s) epoch %d lr %.3e: train_top1 %.2f, valid_top1 %.2f (best_top1 %.2f)',
                        os.environ['SLURM_JOBID'],
                        epoch,
                        current_lr,
                        train_top1_tmp,
                        valid_top1_tmp,
                        best_top1)

        logger.info('(JOBID %s) epoch %d lr %.3e: train_top1 %.2f (best_top1 %.2f)',
                    os.environ['SLURM_JOBID'],
                    epoch,
                    current_lr,
                    train_top1_tmp,
                    best_top1)

    if args.search_type == "cell":
        utils_sparsenas.plot_individual_op_norm(model,
                                                "{}/operator_norm_individual_{}_epoch_{:03d}.png".format(args.path_to_save, run_id, epoch))
    logger.info("args = %s", args)


def train(train_queue, model, criterion, optimizer, lr, report_freq, logger):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

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
            target = target.cuda(non_blocking=True)

            logits = model(input)
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
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.1, help='init momentum')
    parser.add_argument('--search_space', choices=["darts", "s1", "s2", "s4"],
                        default="darts", help="spaces from RobustDARTS; default is DARTS search space")
    parser.add_argument('--weight_decay', type=float, default=45, help='the regularization gain (mu)')
    parser.add_argument('--search_type', choices=[None, "cell"],
                        default="cell", help='cell: search for a single cell structure (normal cell and reduce cell)')
    parser.add_argument('--normalization', choices=["none", "mul", "div"],
                        default="div", help='normalize the regularization (mu) by operation dimension: none, mul or div')
    parser.add_argument('--normalization_exponent', type=float,
                        default=0.5, help='normalization exponent to normalize the weight decay (mu)')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--use_lr_scheduler', action='store_true', default=False, help='use lr scheduler')
    parser.add_argument('--seed', type=int, default=None, help='random seed if NONE')

    parser.add_argument('--path_to_resume', type=str, default=None, help='path to the pretrained model to prune')
    parser.add_argument('--path_to_save', type=str, default=None, help='path to the folder where the experiment will be saved')
    parser.add_argument('--exp_name', type=str, default="log-search", help='experiment name (default None)')
    parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
    parser.add_argument('--train_portion', type=float, default=1, help='portion of training data (if 1, the validation set is the test set)')
    parser.add_argument('--grad_clip', type=float, default=0, help='gradient clipping (set 0 to turn off)')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--cells', type=int, default=8, help='total number of cells')
    parser.add_argument('--data', type=str, default='/home/SSD/ImageNet_raw/', help='location of the data corpus')
    parser.add_argument('--report_freq', type=int, default=0, help='report frequency (set 0 to turn off)')
    parser.add_argument('--plot_freq', type=int, default=1, help='plot (operation norm) frequency (set 0 to turn off)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--workers', type=int, default=24, help='number of workers')

    args = parser.parse_args()

    if args.path_to_save is None:
        args.path_to_save = "{}/{}".format(args.exp_name, args.exp_name)

    if args.normalization == "none":
        args.normalization_exponent = 0

    main(args)
