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
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model_search_multipath import Network
import utils_sparsenas
from genotypes import PRIMITIVES
import random

from ProxSGD_for_groups import ProxSGD
from ProxSGD_for_operations import ProxSGD as ProxSGD_for_operations

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
    sys.exit(1)

  CIFAR_CLASSES = 10
    
  if args.seed is None:
    args.seed = random.randint(0, 10000)
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.cells, criterion)
  model = model.cuda()

  if args.pretrained_model_path is not None:
    model.load_state_dict(torch.load("{}/full_weights".format(args.pretrained_model_path)))
            
  if args.search_type == "operation":
    pass
  else: # search_for_stage or search_for_cell
    network_search = network_params(args.init_channels, args.cells, 4, PRIMITIVES)
    if args.search_type == "stage":
      model_params = utils_sparsenas.group_model_params_by_stage(model, network_search, mu=args.weight_decay)
    elif args.search_type == "cell":
      model_params = utils_sparsenas.group_model_params_by_cell(model, network_search, mu=args.weight_decay)

  utils_sparsenas.plot_individual_op_norm(model, args.pretrained_model_path+"/operator_norm_individual.png", "mul", 0.5)
  if args.search_type == "stage":
    utils_sparsenas.plot_op_norm_across_stages(model_params, args.pretrained_model_path+"/operator_norm_normalized_stage", "mul", 0.5)
  if args.search_type == "cell":
    utils_sparsenas.plot_op_norm_across_cells(model_params, args.pretrained_model_path+"/operator_norm_normalized_cell", "mul", 0.5)



if __name__ == '__main__':
    # search for a single cell structure (normal cell and reduction cell)
    parser = argparse.ArgumentParser("cifar")
    
    parser.add_argument('--search_type', choices=["cell", "stage", "operation"], 
                        default="cell", help='search type: search for cell, search for stage, search for operation (prune operations)')
    parser.add_argument('--normalization', choices=["none", "mul", "div"], 
                        default="mul", help='normalize the regularization (mu) by operation dimension: none, mul or div')
    parser.add_argument('--normalization_exponent', type=float, 
                        default=0.5, help='normalization exponent to normalize the weight decay (mu)')
    parser.add_argument('--weight_decay', type=float, default=0.004, help='weight decay (mu)')
    
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to the pretrained model to prune')
    
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--cells', type=int, default=8, help='total number of cells')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    
    args = parser.parse_args()
        
    if args.normalization == "none":
        args.normalization_exponent = 0
    
    main(args)