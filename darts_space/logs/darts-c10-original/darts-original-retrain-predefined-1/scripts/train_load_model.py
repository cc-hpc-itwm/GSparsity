import torch
import argparse
import genotypes
from model import NetworkCIFAR as Network
import copy

parser = argparse.ArgumentParser("cifar")

parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='darts-retraining', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--pruning_threshold', type=float, default=1e-6, help='operation pruning threshold')
parser.add_argument('--new_initialization', action='store_true', default=False, help='new initialization after pruning')

args = parser.parse_args()

CIFAR_CLASSES = 10

def load_model(filename, args):
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    model.load_state_dict(torch.load(filename))
    return model

def group_model_params(model, reduction_cell_indices, num_edges=14, num_ops=7):
  param_dict_normal = dict()
  param_dict_reduce = dict()
  param_others = []
 
  for cell_index, m in enumerate(model.cells):
        print("cell index is {}".format(cell_index))
        edge_index = 0
        op_index = 0
        subop_index = 0
        
        for name, param in m.named_parameters():
            if "_ops" not in name:
                continue
            
            print(name)
            
            if edge_index <= 9:
                print("name: {}, edge_index is {}".format(name, name[5:6]))
                edge_index_new = int(name[5:6])
            else:
                print("name: {}, edge_index is {}".format(name, name[5:6]))
                edge_index_new = int(name[5:7])

            if edge_index_new == edge_index: # same edge
                pass
            else: 
                edge_index += 1 # a new edge
            
            if edge_index <= 9:
                op_index = int(name[10]) #example: "3" as in cells.4._ops.9._ops.3.op.1.weight               
                subop_index = int(name[15])
            else:
                op_index = int(name[11]) #example: "3" as in cells.4._ops.13._ops.3.op.1.weight
                subop_index = int(name[16])
            
            key_op = "cells.{}._ops.{}._ops.{}.op.{}".format(cell_index, edge_index, op_index, subop_index)
#             logging.info(key_op)
            
            if key_op in param_dict_normal:
                param_dict_normal[key_op].append(param)
            else:
                param_dict_normal[key_op] = [param]
                
                
#             if "_ops" in name:
#                 if "_ops.0._ops.0" in name:
#                     cur_name = name[0:13]
#                     pre_name = name[0:13]
#                 else:
#                     if edge_index <= 9:
#                         cur_name = name[0:13]
#                     else:
#                         cur_name = name[0:14]
                
#                 if cur_name == pre_name: #still the same op
#                     pass
#                 else:
#                     op_index += 1
#                     if op_index == num_ops:
#                         op_index = 0
#                         edge_index += 1
#                     else:
#                         pre_name = cur_name

#                 if edge_index <= 9:
#                     cur_name = name[0:13]
#                 else:
#                     cur_name = name[0:14]
                           
# #                 logging.info("  name is   {}, edge index is {}, op index is {}".format(name, edge_index, op_index))
# #                 logging.info("  cur_name: {}".format(cur_name))
                
#                 if cell_index in reduction_cell_indices:
#                     param_dict_reduce[cur_name].append(param)
#                 else:
#                     param_dict_normal[cur_name].append(param)
#             else:
#                 param_others.append(param)

#   model_params = []

#   for key, value in param_dict_normal.items():
#     model_params.append(dict(params=value, label="normal", op_name=key))
    
#   for key, value in param_dict_reduce.items():
#     model_params.append(dict(params=value, label="reduce", op_name=key))
  
#   model_params.append(dict(params=param_others, label="others"))
  
#   return model_params


if __name__ == "__main__":
    file_pretrained_model = "darts-pruning/darts-pruning-lr_0.001_edec_0_rho_0.6_rdec_0_mu_0.005_time_20201209-221957/final_weights_at_epoch_599"
    model = load_model(file_pretrained_model, args)
    group_model_params(model, reduction_cell_indices=[2,5], num_edges=14, num_ops=7)