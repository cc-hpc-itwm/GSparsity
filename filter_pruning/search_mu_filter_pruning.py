import argparse
import os
from os import path, makedirs
import shutil
import time
import json
import logging
import csv
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.resnet as resnet
import models.densenet as densenet201
import numpy as np
import optuna
from optuna.samplers import NSGAIISampler
from ProxSGD_for_filters import ProxSGD
from trainer import train, validate
from prune_utils import prune_filters_based_on_percentage, prune_filters_based_on_threshold
from dataset import load_cifar100, load_mnist, load_cifar10
from utils import set_logger, get_param_vec, compute_cdf, plot_learning_curve, print_nonzeros, save_checkpoint, acc_n_loss, Cutout


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--dataset', dest='dataset',
                    help='datasets from CIFAR10, CIFAR100, MNIST',
                    default='CIFAR10', type=str)
parser.add_argument('--retrain_optimizer', dest='retrain_optimizer',
                    help='Retrain optimizer from adam,adamw,sgd',
                    default='adam', type=str)
parser.add_argument('--network', dest='network',
                    help='networks from resnet, densenet, mlp',
                    default='resnet', type=str)
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--retrain_epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to retrain')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--lr_scheduler', action='store_true', default=True, help='use learning rate scheduler')
parser.add_argument('--training', action='store_true', default=True, help='for Training')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='regularization gain mu')
parser.add_argument('--weight_reg', default=None, type=float, metavar='M',
                    help='weight_reg')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--prune_filters_based_on_percentage', action='store_true', default=False, 
                    help='True if prune filters based on percetage, False if prune filters based on threshold')
parser.add_argument('--pruning_threshold', '--th', default=1e-2, type=float,
                    metavar='TH', help='pruning threshold (default: 1e-8)')
parser.add_argument('--pruning_filters_percentage', '--pt', default=0, type=float,
                    metavar='PT', help='percentage of filters pruned (default: 0)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--trained_model_path', default='', type=str, metavar='PATH',
                    help='path to trained model to retrain (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='search_mu_filter_pruning', type=str)
parser.add_argument('--filename', dest='filename',
                    help='filename used to save the trained models',
                    default='checkpoint.pth.tar', type=str)
parser.add_argument('--arg_filename', dest='arg_filename',
                    help='filename used to save the arguments of experimet',
                    default='experiment_args.txt', type=str)
parser.add_argument('--csv_filename', dest='csv_filename',
                    help='csv filename used to save the accuracies and compression rate',
                    default='accuracy_comparison_basedon_thresholds.csv', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--n_trials', type=int, default=2, help='Number of Trials for Search')
parser.add_argument('--study_name', dest='study_name',
                    help='study name for hyperparameter search',
                    type=str, default='hyperparameter_search_filter_pruning_mu')

# This function is used for training and retraining and it returns best model's path (model_path) and its accuracy (prec1).
def train_model(model, num_epochs, optimizer, lr_scheduler, run_id, result_dir, logger, original_model_acc1, pruned_model_acc1, pruned_threshold, retrain):

    test_arr1 = []
    test_loss_arr = []
    train_acc_arr = []
    train_loss_arr = []
    lossarr = []
    l1_loss = []
    best_prec1 = 0
    
    epochs_since_improvement = 0
    
    #training/retraining
    if retrain == True:
        train_dir = "{}/retrain/{}/".format(result_dir,str(pruned_threshold))
        print("\nRetraining for Pruning Threshold :", pruned_threshold)
    else:
        train_dir = "{}/train/".format(result_dir)
    
    if not path.exists(path.dirname(train_dir)):   # Check if folder/Path for experiment result exists
        makedirs(path.dirname(train_dir))          # If not, then create one   
    
    if args.training == False:
        with open(train_dir + args.arg_filename, "w") as f:
            json.dump(args.__dict__, f, indent=2)
        
        logger = set_logger(logger_name=train_dir + "logging_" + run_id)
        logger.info("run_id: {}".format(run_id))
  
    current_loss, prec1, filter_compression_rate, weight_compression_rate, weight_percentage_pruned = validate(val_loader, model, criterion, logger, args)
    test_arr1.append(prec1)
    test_loss_arr.append(current_loss)
    
    for epoch in range(args.start_epoch, num_epochs):
        if epochs_since_improvement == 200:
            break
            
        print("current lr {:.5e}".format(optimizer.param_groups[0]["lr"]))

        # train for one epoch
        train_acc, train_loss, loss, l1it = train(train_loader, model, criterion, optimizer, epoch, args.weight_reg, logger, args, retrain)

        lossarr.append(loss)
        l1_loss.append(l1it)
        train_acc_arr.append(train_acc)
        train_loss_arr.append(train_loss)
        
        if args.lr_scheduler:
            lr_scheduler.step()
        
        # evaluate on validation set
        current_loss, prec1, filter_compression_rate, weight_compression_rate, weight_percentage_pruned = validate(val_loader, model, criterion, logger, args)
        test_arr1.append(prec1)
        test_loss_arr.append(current_loss)
        
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
          
        np.save(train_dir + "train_loss_" + run_id,train_loss_arr)
        np.save(train_dir + "train_acc_" + run_id,train_acc_arr)
        np.save(train_dir + "test_acc1_" + run_id,test_arr1)
        np.save(train_dir + "test_loss_" + run_id,test_loss_arr)

        acc_n_loss(train_loss_arr, test_arr1, "{}/acc_n_loss_{}.png".format(train_dir, run_id), train_acc_arr, test_loss_arr)
        save_checkpoint(epoch, epochs_since_improvement, args, model, optimizer, is_best, current_loss, best_prec1, train_dir, original_model_acc1, pruned_model_acc1, pruned_threshold, run_id)
        plot_learning_curve(model, data=model, xlabel="L2 Norm Value of Filters", ylabel="CDF", filename=train_dir + "plot_cdf_" + run_id + ".png", cdf_data=True)
    
        
    model_path = train_dir + "BEST_" + args.filename   # saving path for best model 
    #save loss and accuracies in numpy files
    lossarr = np.hstack(lossarr)
    np.save(train_dir + "loss_" + run_id, lossarr)
    np.save(train_dir + "L1_" + run_id, l1_loss)
    np.save(train_dir + "acc1_" + run_id, test_arr1)
    
    #plot sparsity, training loss, top-1 accuracy and top-5 accuracy
    plot_learning_curve(model, data=model, xlabel="L2 Norm Valueof Filters", ylabel="CDF", filename=train_dir + "plot_cdf_" + run_id + ".png", cdf_data=True)
    plot_learning_curve(model, data=np.log(lossarr), xlabel="Iteration", ylabel="Training Loss", filename=train_dir + "plot_loss_" + run_id + ".png")
    plot_learning_curve(model, data=test_arr1, xlabel="Epoch", ylabel="Top-1 Accuracy", filename=train_dir + "plot_acc1_" + run_id + ".png", ylim=[0, 100])
    
    return model_path


def main(trial, args):
    #Build model
    if args.network == "densenet":
        model = densenet201()
    elif args.network == "mlp":
        model= MLP()
    elif args.network == "resnet":
        model=resnet.resnet56()

    model.cuda()
    cudnn.benchmark = True
    
    lr = 0.0007701
    momentum = 0.5762
    weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.1 , log=True)
    
    run_id = "lr_{}_momentum_{}_weight_decay_{}".format(lr, momentum, weight_decay)
        
    result_dir = "{}/experiment_{}_{}_{}/".format(args.save_dir, args.network, args.dataset, run_id)

    # Check the result_dir exists or not
    if not path.exists(path.dirname(result_dir)):   # Check if folder/Path for experiment result exists
        makedirs(path.dirname(result_dir))          # If not, then create one   
        
    #save arguments to json file
    if args.training:
        with open(result_dir+args.arg_filename, "w") as f:
            json.dump(args.__dict__, f, indent=2)

        #Set logger
        logger = set_logger(logger_name="{}logging_{}".format(result_dir, run_id))
        logger.info("lr: {}, momentum: {}, weight_decay: {}".format(lr, momentum, weight_decay))
    
        optimizer = ProxSGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=0.00001)
    
        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer = checkpoint["optimizer"]
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
                
        #Train the model using ProxSGD
        trained_model_path = train_model(model, args.epochs, optimizer, lr_scheduler, run_id, result_dir,logger, 0, 0, 0, retrain=False)
    
    else:
        #Use Trained model saved at trained_model_path
        trained_model_path = args.trained_model_path
        logger = set_logger(logger_name=result_dir+"logging_retrain"+run_id)

    checkpoint = torch.load(trained_model_path)
    model.load_state_dict(checkpoint["state_dict"])
    
    print("\nOriginal Model Evaluation: ")
    original_model_loss, original_model_acc1, filter_compression_rate, weight_compression_rate, weight_percentage_pruned = validate(val_loader, model, criterion, logger, args)
    
    #Prune filters based on percentage or given threshold
    if prune_filters_based_on_percentage == True:
        print("\nPruned Model Evaluation for filter percentage: ", args.pruning_filters_percentage)
        pruned_model = prune_filters_based_on_percentage(model, args.pruning_filters_percentage) #Return model after pruning specific % of filters
    else:
        print("\nPruned Model Evaluation for Threshold value: ", args.pruning_threshold)
        pruned_model = prune_filters_based_on_threshold(model, args.pruning_threshold) #Return model after pruning specific % of filters
    
    #Model evaluation after pruning
    pruned_model_loss, pruned_model_acc1, filter_compression_rate, weight_compression_rate, weight_percentage_pruned = validate(val_loader, pruned_model, criterion, logger, args) 
    
    # Set optimizer for retraining
    if args.retrain_optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.retrain_optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer= torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.5, 0.99), weight_decay=args.weight_decay)
    
    #Set LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.retrain_epochs), eta_min=0.00001)
    
    #Retrain model 
    retrained_model_path = train_model(pruned_model, args.retrain_epochs, optimizer, lr_scheduler, run_id, result_dir, logger, original_model_acc1, pruned_model_acc1, args.pruning_threshold, retrain=True)
    
    #Model evaluation after retraining
    checkpoint = torch.load(retrained_model_path)
    model.load_state_dict(checkpoint["state_dict"])
    
    print("\nRetrained Model Evaluation for Threshold value: ", args.pruning_threshold)
    retrained_model_loss, retrained_model_acc1, filter_compression_rate, weight_compression_rate, weight_percentage_pruned = validate(val_loader, model, criterion, logger, args)

    result_accuracies_for_each_run = [run_id]
    result_accuracies_for_each_run.append(float("{:.3f}".format(weight_decay)))
    result_accuracies_for_each_run.append(float("{:.3f}".format(original_model_acc1)))
    result_accuracies_for_each_run.append(float(args.pruning_threshold))
    result_accuracies_for_each_run.append(float("{:.3f}".format(retrained_model_acc1)))
    result_accuracies_for_each_run.append(float("{:.3f}".format(weight_percentage_pruned)))
    result_accuracies_for_each_run.append(filter_compression_rate)    
    result_accuracies_for_each_run.append(weight_compression_rate)
        
    with open(args.save_dir+"/"+args.csv_filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow(result_accuracies_for_each_run)
        f.close()
    
    return original_model_acc1, retrained_model_acc1, weight_percentage_pruned


if __name__ == "__main__":
    global args
    args = parser.parse_args()
    
    #load Dataset
    if args.dataset == "CIFAR10":
        train_loader, val_loader = load_cifar10(args.batch_size,args.workers,args.cutout,args.cutout_length)
    elif args.dataset == "CIFAR100":
        train_loader, val_loader = load_cifar100(args.batch_size,args.workers,args.cutout,args.cutout_length)
    elif args.dataset == "MNIST":
        train_loader, val_loader = load_mnist()
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if not path.exists(args.save_dir):   # Check if folder/Path for experiment result exists
        makedirs(args.save_dir)          # If not, then create one

    with open(args.save_dir+"/"+args.csv_filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow(["Run_id", "weight_decay", "Original_Accuracy", "Pruned_threshold", "Retrain Accuracy", "weight_percentage_pruned", "filter_compression_rate", "weight_compression_rate"])  # Write column names to the csv file.
        f.close()
        
    #Create study and storage to save history of hyperparameter search using optuna library
    storage = optuna.storages.RDBStorage(url="sqlite:///"+args.study_name+".db",
                                         engine_kwargs={"pool_pre_ping": True,"connect_args": {"timeout": 10}})
    study = optuna.create_study(study_name=args.study_name, 
                                storage=storage,
                                directions=["maximize", "maximize","maximize"],
                                load_if_exists=True,
                                sampler=NSGAIISampler())
    study.optimize(lambda trial: main(trial, args), n_trials=args.n_trials, gc_after_trial=True)
       
