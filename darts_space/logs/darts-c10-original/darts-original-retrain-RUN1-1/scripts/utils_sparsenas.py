import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model_search_multipath import Network as SearchNetwork
import copy
from genotypes import PRIMITIVES

def plot_individual_op_norm(model, filename, figsize_width=150, normalize=False):
    #generate a sparsity plot of a given network
    Num_pars = [torch.norm(p) for p in model.parameters() if p.requires_grad]
    names = [q for q,p in model.named_parameters() if p.requires_grad]
    par_layers = [p.numel() for p in model.parameters() if p.requires_grad]
    Num_pars = torch.tensor(Num_pars)

    
#     for i in range(10):
#         print(Num_pars[i])

    f = plt.figure(num=None, figsize=(figsize_width, 6), dpi=100, facecolor='w', edgecolor='k')
    if normalize:
        denomi = torch.tensor(np.sqrt(par_layers))
        for i, par in enumerate(Num_pars):
            plt.semilogy(i,(par/denomi[i]).cpu().detach().numpy(),"o")
    else:
        for i, par in enumerate(Num_pars):
            plt.semilogy(i,par.cpu().detach().numpy(),"o")

    my_xticks = names
    plt.xticks(np.arange(len(names)), my_xticks,rotation=90)
    plt.xlim(0,len(names))
    
    plt.ylim(1e-6, 1e2)
    
    plt.tight_layout()
    plt.grid(True)

    plt.savefig(filename)
    plt.close()


def acc_n_loss(train_loss, test_top1, filename, train_top1=None, test_loss=None):
    if train_top1 is not None and test_loss is not None:
        fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.2))
        fig.suptitle('Loss and Acc')
        axs[0,0].semilogy(train_loss)
        axs[0,0].grid(True)
#         axs[0,0].title.set_text('Loss')
        axs[0,0].set_xlabel('Epochs')
        axs[0,0].set_ylabel('Training loss')
#         axs[0,0].set_xticks(np.arange(0, len(train_loss)+1, 100))

        axs[0,1].plot(train_top1)
        axs[0,1].grid(True)
        axs[0,1].set_ylim(0,101)
        axs[0,1].set_yticks(np.arange(0, 101, 5))
        axs[0,1].set_xlabel('Epochs')
        axs[0,1].set_ylabel('Train accuracy (in %)')
#         axs[0,1].set_xticks(np.arange(0, len(train_top1)+1, 100))
#         axs[0,1].title.set_text('Accuracy')

        axs[1,0].semilogy(test_loss)
        axs[1,0].grid(True)
        axs[1,0].set_xlabel('Epochs')
        axs[1,0].set_ylabel('Test loss')
#         axs[1,0].set_xticks(np.arange(0, len(test_top1)+1, 100))
#         axs[1,0].title.set_text('Accuracy')

        axs[1,1].plot(test_top1)
        axs[1,1].grid(True)
        axs[1,1].set_ylim(0,101)
        axs[1,1].set_yticks(np.arange(0, 101, 5))
        axs[1,1].set_xlabel('Epochs')
        axs[1,1].set_ylabel('Test accuracy (in %)')
#         axs[1,1].set_xticks(np.arange(0, len(test_top1)+1, 100))
#         axs[1,1].title.set_text('Accuracy')

        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)       
        
    elif train_top1 is not None and test_loss is None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 4.8))
        fig.suptitle('Loss and Acc')
        ax1.semilogy(train_loss)
        ax1.grid(True)
#         ax1.title.set_text('Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training loss')
#         ax1.set_xticks(np.arange(0, len(train_loss)+1, 100))

        ax2.plot(train_top1)
        ax2.grid(True)
        ax2.set_ylim(0,101)
        ax2.set_yticks(np.arange(0, 101, 5))
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Train accuracy (in %)')
#         ax2.set_xticks(np.arange(0, len(train_top1)+1, 100))
#         ax2.title.set_text('Accuracy')

        ax3.plot(test_top1)
        ax3.grid(True)
        ax3.set_ylim(0,101)
        ax3.set_yticks(np.arange(0, 101, 5))
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Test accuracy (in %)')
#         ax3.set_xticks(np.arange(0, len(test_top1)+1, 100))
#         ax3.title.set_text('Accuracy')

        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
    elif train_top1 is None and test_loss is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 4.8))
        fig.suptitle('Loss and Acc')
        ax1.semilogy(train_loss)
        ax2.plot(test_top1)
        ax1.grid(True)
        ax2.grid(True)
        #ax1.set_ylim(bottom=0)
        ax2.set_ylim(0,101)
        ax2.set_yticks(np.arange(0, 101, 5))
        ax1.title.set_text('Loss')
        ax1.set_xlabel('Epochs')
        ax2.set_xlabel('Epochs')
        ax1.set_ylabel('Training loss')
        ax2.set_ylabel('Test accuracy (in %)')
        ax2.title.set_text('Accuracy')
        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

def num_modpars_cells(model,thresh=0.1,method='None'):
    methods = ['None','Multiply','Divide']
    assert method in methods
    
    num_cells = 0
    num_ks=0
    num_is=0
    num_ls = 0
    num_cellops = 0
    allpars = 0
    for cell in model.cells:
        for i,op in enumerate(cell._ops):
            for k,b in enumerate(op.children()):
                for l,n in enumerate(b.children()):
                    fulli = sum(p.numel() for p in n.parameters() if p.requires_grad)
                    par_layers = [p.numel() for p in n.parameters() if p.requires_grad]
                    Num_pars = [torch.norm(p) for p in n.parameters() if p.requires_grad]

                    if len(Num_pars)==0:
                        continue
                    if method == 'None':
                        scld = torch.stack(Num_pars)
                    elif method == 'Multiply':
                        scld = torch.stack(Num_pars)*torch.tensor(np.sqrt(par_layers)).cuda()
                    elif method == 'Divide':
                        scld = torch.stack(Num_pars)/torch.tensor(np.sqrt(par_layers)).cuda()
                    if (scld < thresh).any():
                        for p in n.parameters():
                            p.data = torch.zeros_like(p)
                            p.requires_grad = False
                    else:
                        num_cellops +=1
                    num_ls+=1
                num_ks+=1
                #print(b)
            num_is += 1
        num_cells+=1
    print(num_cellops,num_ls,num_ks,num_is,num_cells)
    
    return num_cellops


def num_modpars(model,thresh=0.1,method='None'):
    methods = ['None','Multiply','Divide']
    assert method in methods
        
    num_cellops = 0
    allpars = 0
    for cell in model.cells:
        for k,b in enumerate(cell.preprocess0.children()):
            for l,n in enumerate(b.children()):
                #print("N",n)
                fulli = sum(p.numel() for p in b.parameters() if p.requires_grad)
                par_layers = [p.numel() for p in b.parameters() if p.requires_grad]
                Num_pars = [torch.norm(p) for p in b.parameters() if p.requires_grad]
                if len(Num_pars)==0:
                    continue
                if method == 'None':
                    scld = torch.stack(Num_pars)
                elif method == 'Multiply':
                    scld = torch.stack(Num_pars)*torch.tensor(np.sqrt(par_layers)).cuda()
                elif method == 'Divide':
                    scld = torch.stack(Num_pars)/torch.tensor(np.sqrt(par_layers)).cuda()
                if (scld < thresh).any():
                    for p in b.parameters():
                        p.data = torch.zeros_like(p)
                        p.requires_grad = False

        for k,b in enumerate(cell.preprocess1.children()):
            for l,n in enumerate(b.children()):
                #print("N",n)
                fulli = sum(p.numel() for p in b.parameters() if p.requires_grad)
                par_layers = [p.numel() for p in b.parameters() if p.requires_grad]
                Num_pars = [torch.norm(p) for p in b.parameters() if p.requires_grad]
                if len(Num_pars)==0:
                    continue
                if method == 'None':
                    scld = torch.stack(Num_pars)
                elif method == 'Multiply':
                    scld = torch.stack(Num_pars)*torch.tensor(np.sqrt(par_layers)).cuda()
                elif method == 'Divide':
                    scld = torch.stack(Num_pars)/torch.tensor(np.sqrt(par_layers)).cuda()
                if (scld < thresh).any():
                    for p in b.parameters():
                        p.data = torch.zeros_like(p)
                        p.requires_grad = False
        
        for i,op in enumerate(cell._ops):
            for k,b in enumerate(op.children()):
                for l,n in enumerate(b.children()):
                    fulli = sum(p.numel() for p in n.parameters() if p.requires_grad)
                    par_layers = [p.numel() for p in n.parameters() if p.requires_grad]
                    Num_pars = [torch.norm(p) for p in n.parameters() if p.requires_grad]
                    if len(Num_pars)==0:
                        continue
                    if method == 'None':
                        scld = torch.stack(Num_pars)
                    elif method == 'Multiply':
                        scld = torch.stack(Num_pars)*torch.tensor(np.sqrt(par_layers)).cuda()
                    elif method == 'Divide':
                        scld = torch.stack(Num_pars)/torch.tensor(np.sqrt(par_layers)).cuda()
                    if (scld < thresh).any():
                        for p in n.parameters():
                            p.data = torch.zeros_like(p)
                            p.requires_grad = False
                    else:
                        num_cellops +=1

    
    par_layers = [p.numel() for p in model.classifier.parameters() if p.requires_grad]
    Num_pars = [torch.norm(p) for p in model.classifier.parameters() if p.requires_grad]
    if len(Num_pars)!=0:
        if method == 'None':
            scld = torch.stack(Num_pars)
        elif method == 'Multiply':
            scld = torch.stack(Num_pars)*torch.tensor(np.sqrt(par_layers)).cuda()
        elif method == 'Divide':
            scld = torch.stack(Num_pars)/torch.tensor(np.sqrt(par_layers)).cuda()

        if (scld[0] < thresh).any():
            for q,p in model.classifier.named_parameters():
                p.data = torch.zeros_like(p)
                p.requires_grad = False
    
    return num_cellops

def group_model_params_by_cell(model, network, mu=None):
    #This functions put the same operation of different cells into the same group (the group will be passed to optimizer)
  param_dict_normal = dict()
  param_dict_reduce = dict()
  param_others = []

  for edge in range(network.num_edges):
    for op in range(network.num_ops):
        param_dict_normal["_ops.{}._ops.{}".format(edge, op)] = []
        param_dict_reduce["_ops.{}._ops.{}".format(edge, op)] = []
  
  for op in model.stem:
        for param in op.parameters():
            param_others.append(param)
                
#   model.global_pooling is not trainable
  classifier_weight, classifier_bias = model.classifier.parameters()
  param_others.extend([classifier_weight, classifier_bias])
  
  for cell_index, m in enumerate(model.cells):
#         print("cell index is {}".format(cell_index))
        op_index = 0
        edge_index = 0
        
        for name, param in m.named_parameters():
            if "_ops" in name:
                if "_ops.0._ops.0" in name: # beginning of a new cell
                    cur_name = name[0:13] # assuming the number of cells < 10
                    pre_name = name[0:13]
                else:
                    if edge_index <= 9:
                        cur_name = name[0:13] #example: extract "_ops.3._ops.4" from "_ops.3._ops.4.op.2.weight"
                    else:
                        cur_name = name[0:14] #example: extract "_ops.13._ops.4" from "_ops.13._ops.4.op.2.weight"
                
                if cur_name == pre_name: #still the same op
                    pass
                else:
                    op_index += 1
                    if op_index == network.num_ops:
                        op_index = 0
                        edge_index += 1
                    else:
                        pre_name = cur_name

                if edge_index <= 9:
                    cur_name = name[0:13]
                else:
                    cur_name = name[0:14]
                           
#                 print("  name is   {}, edge index is {}, op index is {}".format(name, edge_index, op_index))
#                 print("  cur_name: {}".format(cur_name))
                
                if cell_index in network.reduce_cell_indices:
                    param_dict_reduce[cur_name].append(param)
#                     if cur_name in param_dict_reduce:
#                         print("{} already in param_dict_reduce: {}".format(cur_name, param_dict_reduce[cur_name]))
                else:
                    param_dict_normal[cur_name].append(param)
#                     if cur_name in param_dict_normal:
#                         print("{} already in param_dict_normal: {}".format(cur_name, param_dict_normal[cur_name]))
            else:
                param_others.append(param)

  model_params = []
  for key, value in param_dict_normal.items():
    model_params.append(dict(params=value, label="normal", op_name=key, weight_decay=mu))    
  for key, value in param_dict_reduce.items():
    model_params.append(dict(params=value, label="reduce", op_name=key, weight_decay=mu))  
  model_params.append(dict(params=param_others, label="unprunable", op_name="unprunable", weight_decay=None))
  
  return model_params

def group_model_params_by_stage(model, network, mu=None):
    #This functions put the same operation in different cells in the same stage into the same group (Example: stage normal 1, stage reduce 1, stage normal 2, stage reduce 2, stage normal 3. Each stage consists of several cells. Here normal cells and reduce cells are not differentiated.)
  param_others = []

  for op in model.stem:
        for param in op.parameters():
            param_others.append(param)
                
#   model.global_pooling is not trainable
  classifier_weight, classifier_bias = model.classifier.parameters()
  param_others.extend([classifier_weight, classifier_bias])
  
  param_dict_cell = {}
  stage_normal = 1
  stage_reduce = 1
  for cell_index, m in enumerate(model.cells):
        for name, param in m.named_parameters():
            if "_ops" in name:                                   
                if cell_index in network.reduce_cell_indices: #reduction cell
                    key_op = "stage_reduce_{}.{}".format(stage_reduce, name)
                else: #normal cell
                    key_op = "stage_normal_{}.{}".format(stage_normal, name)

                if key_op in param_dict_cell:
#                     logging.info("key is in dictionary.")
                    param_dict_cell[key_op].append(param)
                else:
#                     logging.info("key is NOT in dictionary!")
                    param_dict_cell[key_op] = [param]
            else:
                param_others.append(param)
                        
        if cell_index in network.reduce_cell_indices:
            stage_normal += 1
            stage_reduce += 1

  model_params = []
  for key, value in param_dict_cell.items():
    model_params.append(dict(params=value, label="prunable", name=key, weight_decay=mu))    
  model_params.append(dict(params=param_others, label="unprunable", name="unprunable", weight_decay=None))
  
  return model_params


def compute_op_norm_across_cells(model_params):
    # compute the norm of the vector containing the weights of the same operation in different cells (e.g., sep_conv_3x3)
    # normal cells and reduction cells are computed separately
    op_norm_normal_dict = {}
    op_norm_reduce_dict = {}
    for operation in model_params:
        if operation["label"] == "unprunable": # weights from ops like stem, cell preprocessing and classifier.
            continue
        
        params = operation["params"]
        params_norm_square = 0
        for param in params:
            params_norm_square += torch.norm(param) ** 2 
         
        if operation["label"] == "normal":
            op_norm_normal_dict[operation["op_name"]] = torch.sqrt(params_norm_square) # take the square root to get the L2 norm
        elif operation["label"] == "reduce":
            op_norm_reduce_dict[operation["op_name"]] = torch.sqrt(params_norm_square)

    return op_norm_normal_dict, op_norm_reduce_dict

def plot_op_norm_across_cells(model_params, filename):
    op_norm_normal_dict, op_norm_reduce_dict = compute_op_norm_across_cells(model_params)
        
    f1 = plt.figure(num=None, figsize=(98*0.15, 6), dpi=100, facecolor='w', edgecolor='k')    
    num_ops = 0
    op_names = []
    for op_name, op_norm in op_norm_normal_dict.items():
        op_names.append(op_name)
        plt.semilogy(num_ops, (op_norm).cpu().detach().numpy(), "o")
        num_ops += 1
    plt.xticks(np.arange(num_ops), op_names, rotation=90)
    plt.xlim(0,num_ops)
    plt.ylim(1e-6, 1e2)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("{}_normal.png".format(filename))
    plt.close()
    
    f2 = plt.figure(num=None, figsize=(98*0.15, 6), dpi=100, facecolor='w', edgecolor='k')    
    num_ops = 0
    op_names = []
    for op_name, op_norm in op_norm_reduce_dict.items():
        op_names.append(op_name)
        plt.semilogy(num_ops, (op_norm).cpu().detach().numpy(), "o")
        num_ops += 1
    plt.xticks(np.arange(num_ops), op_names, rotation=90)
    plt.xlim(0, num_ops)
    plt.ylim(1e-6, 1e2)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("{}_reduce.png".format(filename))
    plt.close()
    
def compute_op_norm_across_stages(model_params):
    op_norm_dict = {}
    for operation in model_params:
        if operation["label"] == "unprunable":
            continue
        
        params = operation["params"]
        params_norm_square = 0
        for param in params:
            params_norm_square += torch.norm(param) ** 2
         
        op_norm_dict[operation["name"]] = torch.sqrt(params_norm_square)
    
    return op_norm_dict

def plot_op_norm_across_stages(model_params, filename):
    op_norm_dict = compute_op_norm_across_stages(model_params)
        
    f = plt.figure(num=None, figsize=(10*98*0.15, 6), dpi=100, facecolor='w', edgecolor='k')
    num_ops = 0
    op_names = []
    for op_name, op_norm in op_norm_dict.items():
        op_names.append(op_name)
        plt.semilogy(num_ops, (op_norm).cpu().detach().numpy(), "o")
        num_ops += 1
#     print("The number of ops is {}".format(num_ops))
    
    plt.xticks(np.arange(num_ops), op_names,rotation=90)
    plt.xlim(0,num_ops)

    plt.ylim(1e-6, 1e2)
    
    plt.tight_layout()
    plt.grid(True)

    plt.savefig(filename)
    plt.close()


def get_viz(model):
  all_cells = []
  for cell in model.cells:
    cell_struct = []
    for i,op in enumerate(cell._ops):
      for k,b in enumerate(op.children()):
        inside = []
        for l,n in enumerate(b.children()):
          flag = False
          for p in n.parameters():
            if not p.requires_grad:
              flag = True
          if flag:
            inside.append(0)
          else:
            inside.append(1)
        #print(i,k,b)
      cell_struct.append(inside)
    #print(np.vstack(cell_struct))
#     all_cells.append((cell.reduction,cell_struct))
    all_cells.append((cell.reduction,np.vstack(cell_struct)))
  #print(all_cells)
  return all_cells

def discretize_search_model_by_cell(model_path, network_eval, network_search, threshold, CIFAR_CLASSES = 10):
# removing the ops with a small norm and the discrete cell will be scaled up for evaluation

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = SearchNetwork(network_search.init_channels, CIFAR_CLASSES, network_search.cells, criterion)
  model = model.cuda()
  model.load_state_dict(torch.load(model_path))

  # these two variables may change when the number of intermediate nodes and candidate operations change
  model_params = group_model_params_by_cell(model, network_search)
  op_norm_normal, op_norm_reduce = compute_op_norm_across_cells(model_params)

  alpha_normal = []
  alpha = []
  edge_index = 0  
  for index, (op_name, op_norm) in enumerate(op_norm_normal.items()):        
        if  edge_index * network_search.num_ops <= index < (edge_index + 1) * network_search.num_ops:            
#             print("index {} (edge {}): op {}, norm {}".format(index, edge_index, op_name, op_norm))
            if op_norm <= threshold:
                alpha.append(0)
            else:
                alpha.append(1)
            if index == (edge_index + 1) * network_search.num_ops - 1:
#                 print("edge {}: alpha {}".format(edge_index, alpha))
                alpha_normal.append(alpha)
                alpha = []
                edge_index += 1
  alpha_normal = torch.tensor(alpha_normal)
#   print("alpha_normal: {}".format(alpha_normal))

  alpha_reduce = []
  alpha = []
  edge_index = 0
  for index, (op_name, op_norm) in enumerate(op_norm_reduce.items()):        
        if  edge_index * network_search.num_ops <= index < (edge_index + 1) * network_search.num_ops:            
#             print("index {} (edge {}): op {}, norm {}".format(index, edge_index, op_name, op_norm))
            if op_norm <= threshold:
                alpha.append(0)
            else:
                alpha.append(1)
            if index == (edge_index + 1) * network_search.num_ops - 1:
#                 print("edge {}: alpha {}".format(edge_index, alpha))
                alpha_reduce.append(alpha)
                alpha = []
                edge_index += 1
  alpha_reduce = torch.tensor(alpha_reduce)
#   print("alpha_reduce: {}".format(alpha_reduce))

  alpha_full = []
  num_reduce_cell = len(network_eval.reduce_cell_indices)
  cur_reduce_cell = 0
  for cell_index in range(network_eval.cells):
    if cell_index < network_eval.reduce_cell_indices[cur_reduce_cell]:
        alpha_full.append((False, np.vstack(alpha_normal)))
    elif cell_index == network_eval.reduce_cell_indices[cur_reduce_cell]:
        alpha_full.append((True,  np.vstack(alpha_reduce)))
        cur_reduce_cell += 1        
        if cur_reduce_cell == num_reduce_cell:
            break            
  for cell_index in range(network_eval.reduce_cell_indices[-1]+1, network_eval.cells):
        alpha_full.append((False, np.vstack(alpha_normal)))
        
  return alpha_full, model.genotype()
        
#   return alphas

def discretize_search_model_by_stage(model_path, network_eval, network_search, threshold=1e-5, CIFAR_CLASSES = 10):
# removing the ops with a small norm and the discrete cell will be scaled up for evaluation

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = SearchNetwork(network_search.init_channels, CIFAR_CLASSES, network_search.cells, criterion)
  model = model.cuda()
  model.load_state_dict(torch.load(model_path))

  # these two variables may change when the number of intermediate nodes and candidate operations change

  model_params = group_model_params_by_stage(model, network_search)
  op_norm_dict = compute_op_norm_across_stages(model_params)

#   print("kkk {}".format(op_norm_dict[next(iter(op_norm_dict))]))
  stage_name = "stage_normal_1"
  edge_index = 0
  op_index = 0
  alpha_edge = [int(op_norm_dict[next(iter(op_norm_dict))].cpu().detach().numpy()>threshold)]
  alpha_stage = []
  alphas = []
  for op_name, op_norm in op_norm_dict.items():
#         print("op_name: {}, norm: {} -> {}".format(op_name, op_norm, op_norm > threshold))
#         print("detected stage {}, edge {}, op {}".format())
        if stage_name == op_name[0:14]: # still in the same stage
            pass
#             print("  still in the same stage")
        else: # in a new stage
            if edge_index == network_search.num_edges-1:
                alpha_stage.append(alpha_edge)
#             print("  in a new stage")
            stage_name = op_name[0:14]
            alphas.append(alpha_stage)
            edge_index = 0 #0<=edge_index<=13
            op_index = 0
            alpha_stage = []
            alpha_edge = [int(op_norm.cpu().detach().numpy() > threshold)]
        
        if edge_index == int(op_name[20: 21 + (edge_index>=10)]): # still in the same edge
            pass
#             print("    in edge {}".format(edge_index))
        else:
            alpha_stage.append(alpha_edge)
            alpha_edge = [1]
            edge_index += 1
            op_index = 0
#             print("    in the new edge {}".format(edge_index))
            
        if op_index == int(op_name[27 + (edge_index>=10)]): # 0<=op_index<=6
#             print("      still in the same op")
            alpha_edge[-1] *= (op_norm.cpu().detach().numpy() > threshold)
        else:
#             print("      in a new op")
            alpha_edge.append(int(op_norm.cpu().detach().numpy() > threshold))
            op_index += 1
  
  alpha_stage.append(alpha_edge)
  alphas.append(alpha_stage)

  alpha_full = []
  num_reduce_cell = len(network_eval.reduce_cell_indices)
  cur_reduce_cell = 0
  for cell_index in range(network_eval.cells):
    if cell_index < network_eval.reduce_cell_indices[cur_reduce_cell]:
        alpha_full.append((False, np.vstack(alphas[2*cur_reduce_cell])))
    elif cell_index == network_eval.reduce_cell_indices[cur_reduce_cell]:
        alpha_full.append((True, np.vstack(alphas[2*cur_reduce_cell+1])))
        cur_reduce_cell += 1        
        if cur_reduce_cell == num_reduce_cell:
            break            
  for cell_index in range(network_eval.reduce_cell_indices[-1]+1, network_eval.cells):
        alpha_full.append((False, np.vstack(alphas[2*cur_reduce_cell])))

#   for _ in range(0, network_eval.cells//3):
#         alpha_full.append((False, np.vstack(alphas[4])))
#   alpha_full.append((True, np.vstack(alphas[1])))
#   for _ in range(network_eval.cells//3+1, (2*network_eval.cells)//3):
#         alpha_full.append((False, np.vstack(alphas[2])))
#   alpha_full.append((True, np.vstack(alphas[3])))
#   for _ in range((2*network_eval.cells)//3+1, network_eval.cells):
#         alpha_full.append((False, np.vstack(alphas[0])))
    
  return alpha_full, model.genotype()