import os
from os import path, makedirs
import argparse
import json
import torch
import torch.nn as nn
import torchvision
import numpy as np
import models.resnet as resnet
import models.resnet_small as resnet_small
from torchsummary import summary
from main_filter_pruning import train_model
from dataset import load_cifar10
from torch.autograd import Variable
import logging
from utils import set_logger, get_param_vec, compute_cdf, plot_learning_curve, print_nonzeros, save_checkpoint, acc_n_loss, Cutout
from trainer import train, validate
from flops_counter import get_model_complexity_info
from prune_utils import prune_filters_based_on_percentage, prune_filters_based_on_threshold

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--dataset', dest='dataset',
                    help='datasets from CIFAR10, CIFAR100, MNIST',
                    default='CIFAR10', type=str)
parser.add_argument('--network', dest='network',
                    help='networks from resnet, densenet, mlp',
                    default='resnet', type=str)
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--retrain_optimizer', dest='retrain_optimizer',
                    help='Retrain optimizer from adam,adamw,sgd',
                    default='adam', type=str)
parser.add_argument('--weight_reg', default=None, type=float, metavar='M',
                    help='weight_reg')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='small_model_thinning', type=str)
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--retrain_epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to retrain')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--lr_scheduler', action='store_true', default=True, help='use learning rate scheduler')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='regularization gain mu')
parser.add_argument('--prune_filters_based_on_percentage', action='store_true', default=False, 
                    help='True if prune filters based on percetage, False if prune filters based on threshold')
parser.add_argument('--pruning_threshold', '--th', default=1e-2, type=float,
                    metavar='TH', help='pruning threshold (default: 1e-8)')
parser.add_argument('--pruning_filters_percentage', '--pt', default=75, type=float,
                    metavar='PT', help='percentage of filters pruned (default: 0)')
parser.add_argument('--training', action='store_true', default=False, help='for Training')
parser.add_argument('--trained_model_path', default='', type=str, metavar='PATH',
                    help='path to trained model to retrain (default: none)')
parser.add_argument('--arg_filename', dest='arg_filename',
                    help='filename used to save the arguments of experimet',
                    default='experiment_args.txt', type=str)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--filename', dest='filename',
                    help='filename used to save the trained models',
                    default='checkpoint.pth.tar', type=str)

# check for zero filters for each layer
def check_channels(tensor):
    num_filters = tensor.size()[0]
    tensor_resize = tensor.view(num_filters, -1)
    filters_if_zero = np.zeros(num_filters)
    for filter in range(0, num_filters):
        filters_if_zero[filter] = np.count_nonzero(tensor_resize[filter].cpu().numpy()) != 0
    
    indices_nonzero = torch.tensor((filters_if_zero != 0).nonzero()[0])
    zeros = (filters_if_zero == 0).nonzero()[0]
    indices_zero =  torch.tensor(zeros) if zeros.size > 0 else []
    return indices_zero, indices_nonzero

# create dictionary with remaining filters in each layer
def create_filters_dict(model, filters_json_filename):
    item = list(model.state_dict().items())

    kept_index_per_layer = {}
    kept_filter_per_layer = {}
    pruned_index_per_layer = {}

    for conv_layer in range(0, len(item)-2, 6):
        indices_zero, indices_nonzero = check_channels(item[conv_layer][1])
        if indices_nonzero.nelement() != 0:
            pruned_index_per_layer[item[conv_layer][0]] = indices_zero
            kept_index_per_layer[item[conv_layer][0]] = indices_nonzero
            kept_filter_per_layer[item[conv_layer][0]] = indices_nonzero.shape[0]
        # if conv1 is present and conv2 is not then set default values for conv2
        elif '.'.join(item[conv_layer][0].split('.')[:-2])+'.conv1.weight' in kept_filter_per_layer.keys():
            if item[conv_layer][0].split('.')[0] == 'layer1':
                kept_index_per_layer[item[conv_layer][0]] = 16
                kept_filter_per_layer[item[conv_layer][0]] = 16
            elif item[conv_layer][0].split('.')[0] == 'layer2':
                kept_index_per_layer[item[conv_layer][0]] = 32
                kept_filter_per_layer[item[conv_layer][0]] = 32
            else:
                kept_index_per_layer[item[conv_layer][0]] = 64
                kept_filter_per_layer[item[conv_layer][0]] = 64
        
    basic_block_flag = list(kept_index_per_layer.keys())
    constrct_flag = basic_block_flag
    block_flag = "conv2"
    # number of nonzero channel in conv1, and four stages
    num_for_construct = []
    out_filters = {}
    out_filters ['layer1'] = {}
    out_filters ['layer2'] = {}
    out_filters ['layer3'] = {}
    out_filters['layer1']['0'] = {}
    out_filters['layer1']['0']['conv1'] = 1
    out_filters['layer1']['0']['conv2'] = 16
    out_filters['layer2']['0'] = {}
    out_filters['layer2']['0']['conv1'] = 1
    out_filters['layer2']['0']['conv2'] = 32
    out_filters['layer3']['0'] = {}
    out_filters['layer3']['0']['conv1'] = 1
    out_filters['layer3']['0']['conv2'] = 64
    
    for idx, key in enumerate(constrct_flag):
        num_for_construct.append(kept_filter_per_layer[key])
        if idx==0:
            #out_filters['conv1'] = kept_filter_per_layer[key]
            continue
        
        stage = key.split('.')[0]
        block = key.split('.')[1]
        layer = key.split('.')[2]

        if block not in out_filters[stage]:
            out_filters[stage][block] = {}
        
        out_filters[stage][block][layer] = {}
        out_filters[stage][block][layer] = kept_filter_per_layer[key]
        
        '''if idx==1 :
            out_filters[stage][block]['in'] = kept_filter_per_layer["conv1.weight"]
            
        if idx%2 == 0:
            if stage == 'layer1' and block == '0':
                continue
            prev_key = constrct_flag[idx-2]
            prev_stage = prev_key.split('.')[0]
            prev_block = prev_key.split('.')[1]
            out_filters[stage][block]['in'] = out_filters[prev_stage][prev_block]['conv2']'''
    
                

         
    with open(filters_json_filename, 'w') as fp:
        json.dump(out_filters, fp, indent=4) 
    
    return kept_index_per_layer, pruned_index_per_layer, block_flag, out_filters

def main(args, result_dir):
 
    model = resnet.resnet56()
    checkpoint = torch.load(args.trained_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    device = "cuda"
    model.to(device)
    #summary(model, (3, 32, 32))
    
    print('*'*35+'Model before thinning'+'*'*35+'\n', model)  
    print('\nModel evaluation before thinning: ')
    current_loss, original_model_acc1, filter_compression_rate,weight_compression_rate, filter_percentage_pruned = validate(val_loader, model, criterion, logger, args)
    
    #Use this commented code to prune the trained model directly without retraining and then apply thinning operation. 
    '''#Prune filters based on percentage or given threshold
    if prune_filters_based_on_percentage:
        pruned_threshold = args.pruning_filters_percentage
        print("\nPruned Model Evaluation for filter percentage: ", pruned_threshold)
        model = prune_filters_based_on_percentage(model, pruned_threshold) #Return model after pruning specific % of filters
    else:
        pruned_threshold = args.pruning_threshold
        print("\nPruned Model Evaluation for filter percentage: ", pruned_threshold)
        model = prune_filters_based_on_threshold(model, pruned_threshold) #Return model after pruning specific % of filters
    
    #Model evaluation after pruning
    pruned_model_loss, pruned_model_acc1, filter_compression_rate, weight_compression_rate, weight_percentage_pruned = validate(val_loader, model, criterion, logger, args) 
    '''
        
    filters_json_filename = '{}/out_filters.json'.format(result_dir)
    
    kept_index_per_layer, pruned_index_per_layer, block_flag, out_filters = create_filters_dict(model, filters_json_filename)
    
    num_blocks = list(len(v.keys()) for k, v in out_filters.items())
    small_model = resnet_small.resnet56(out_filters, num_blocks).to(device)
    print('\n'+'*'*35+'Model after thinning'+'*'*35+'\n', small_model)  
    #summary(model, (3, 32, 32))
    #summary(small_model, (3, 32, 32))
    indice_dict, pruned_index_per_layer, block_flag = kept_index_per_layer, pruned_index_per_layer, block_flag
    #Update state_dict for small model
    big_state_dict = model.state_dict()
    small_state_dict = {}
    keys_list = list(big_state_dict.keys())
    key_list_small = list(indice_dict.keys())
    #print("keys_list", keys_list, key_list_small)
    
    for index, [key, value] in enumerate(big_state_dict.items()):
        # all the conv layer excluding downsample layer
        flag_conv_ex_down = not 'bn' in key and not 'downsample' in key and not 'fc' in key
        if flag_conv_ex_down:
            conv_index = keys_list.index(key)
            if key in indice_dict:
                if key == 'conv1.weight':
                    small_state_dict[key] = value
                    for offset in range(1, 6, 1):
                        bn_key = keys_list[conv_index + offset]
                        small_state_dict[bn_key] = big_state_dict[bn_key]
                        
                
                elif not "conv1" in key:
                    # get the last con layer
                    conv_index_prev = key_list_small.index(key)
                    key_for_input = key_list_small[conv_index_prev - 1]
                    small_state_dict[key] = torch.index_select(big_state_dict[key], 1, indice_dict[key_for_input].to(device))
                    for offset in range(1, 6, 1):
                        bn_key = keys_list[conv_index + offset]
                        small_state_dict[bn_key] = big_state_dict[bn_key]
                
                elif 'conv1' in key:
                    small_state_dict[key] = torch.index_select(big_state_dict[key], 0, indice_dict[key].to(device))
                    for offset in range(1, 6, 1):
                        bn_key = keys_list[conv_index + offset]
                        if 'num_batches_tracked' in bn_key:
                            small_state_dict[bn_key] = big_state_dict[bn_key]
                        else:
                            small_state_dict[bn_key] = torch.index_select(big_state_dict[bn_key], 0, indice_dict[key].to(device))
                                  
            elif key in ['layer1.0.conv1.weight','layer2.0.conv1.weight','layer3.0.conv1.weight']:
                indices = torch.tensor([1])
                small_state_dict[key] = torch.index_select(big_state_dict[key], 0, indices.to(device))
                conv_index = keys_list.index(key)
                for offset in range(1, 6, 1):
                    bn_key = keys_list[conv_index + offset]
                    if 'num_batches_tracked' in bn_key:
                        small_state_dict[bn_key] = big_state_dict[bn_key]
                    else:
                        small_state_dict[bn_key] = torch.index_select(big_state_dict[bn_key], 0, indices.to(device))
            
            elif key in ['layer1.0.conv2.weight','layer2.0.conv2.weight','layer3.0.conv2.weight']:
                indices = torch.tensor([1])
                small_state_dict[key] = torch.index_select(big_state_dict[key], 1, indices.to(device))
                conv_index = keys_list.index(key)
                for offset in range(1, 6, 1):
                    bn_key = keys_list[conv_index + offset]
                    small_state_dict[bn_key] = big_state_dict[bn_key]
                
            elif 'linear' in key:
                small_state_dict[key] = value

    if len(set(big_state_dict.keys()) - set(small_state_dict.keys())) != 0:
        print("different keys of big and small model",
              sorted(set(big_state_dict.keys()) - set(small_state_dict.keys())))
        for x, y in zip(small_state_dict.keys(), small_model.state_dict().keys()):
            if small_state_dict[x].size() != small_model.state_dict()[y].size():
                print("difference with dict and model", x, small_state_dict[x].size(),
                      small_model.state_dict()[y].size())
    for x, y in zip(list(small_state_dict.keys()), list(small_model.state_dict().keys())):
        #small_state_dict[x] = small_model.state_dict()[y]
        small_state_dict[y] = small_state_dict.pop(x)
    
    small_model.load_state_dict(small_state_dict)    
    #summary(small_model, (3, 32, 32))
    torch.save(small_state_dict, result_dir+'/small_model.pth.tar')
    return original_model_acc1, model, small_model

if __name__ == '__main__':
    args = parser.parse_args()
    
    #load Dataset
    if args.dataset=='CIFAR10':
        train_loader, val_loader = load_cifar10(args.batch_size,args.workers,args.cutout,args.cutout_length)
    elif args.dataset=='CIFAR100':
        train_loader, val_loader= load_cifar100(args.batch_size,args.workers,args.cutout,args.cutout_length)
    elif args.dataset=='MNIST':
        train_loader, val_loader = load_mnist()
    
    result_dir = '/'.join(args.trained_model_path.split('/')[:-1])
    run_id = 'logging_thinning'
    logger = set_logger(logger_name=result_dir+"/"+run_id)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    
    if not path.exists(args.save_dir):   # Check if folder/Path for experiment result exists
        makedirs(args.save_dir)          # If not, then create one  
    
    #build small model    
    original_model_acc1, model, small_model = main(args, result_dir)
    print('\nModel evaluation after thinning: ')
    current_loss, pruned_model_acc1, filter_compression_rate,weight_compression_rate, filter_percentage_pruned = validate(val_loader, small_model, criterion, logger, args)
    
    #retrain model after thinning
    if args.retrain_optimizer=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.retrain_optimizer=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optimizer=='adamw':
        optimizer= torch.optim.AdamW(model.parameters(), lr= 0.0001, betas=(0.5, 0.99), weight_decay=args.weight_decay)
    
    lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.retrain_epochs), eta_min=0.00001)

    retrained_model_path,retrained_accuracy = train_model(small_model, args, train_loader, val_loader, optimizer,lr_scheduler,run_id, result_dir,logger, original_model_acc1, pruned_model_acc1, 'after_thinning' ,retrain=True )
    print('\nModel evaluation after thinning and retraining: ')
    checkpoint = torch.load(retrained_model_path)
    small_model.load_state_dict(checkpoint['state_dict'])
    current_loss, final_model_acc1, filter_compression_rate,weight_compression_rate, filter_percentage_pruned = validate(val_loader, small_model, criterion, logger, args)
    print('\n Model FLOPs: ')
    flops, params = get_model_complexity_info(model, (3,32,32), print_per_layer_stat=False)
    flops_compress, params_compress = get_model_complexity_info(small_model, (3,32,32), print_per_layer_stat=False)
    print('flops compressed', flops_compress)
    print('total flops', flops)
    print('params compressed', params_compress)
    print('total params', params)
    print('FLOPs ratio {:.2f} = {:.4f} [G] / {:.4f} [G]; Parameter ratio {:.2f} = {:.2f} [k] / {:.2f} [k].'
          .format(flops_compress / flops * 100, flops_compress / 10. ** 9, flops / 10. ** 9,
                  params_compress / params * 100, params_compress / 10. ** 3, params / 10. ** 3))
    flops_compress, params_compress = get_model_complexity_info(small_model, (3,32,32), print_per_layer_stat=False)