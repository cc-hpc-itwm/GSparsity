import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model_search_multipath import Network as SearchNetwork
from graphviz import Digraph

def plot_individual_op_norm(model, filename, normalization=False, normalization_exponent=0):
    """plot the norm of each operation in the given model"""
    
    op_norms = [torch.norm(p) for p in model.parameters() if p.requires_grad]
    op_names = [q for q, p in model.named_parameters() if p.requires_grad]
    op_sizes = [p.numel() for p in model.parameters() if p.requires_grad]
        
    num_ops = len(op_names)
    f = plt.figure(num=None, figsize=(num_ops*0.15, 6), dpi=100, facecolor='w', edgecolor='k')
    if normalization:
        normalized_op_sizes = torch.pow(op_sizes, normalization_exponent)
        for i, op_norm in enumerate(op_norms):
            plt.semilogy(i, (op_norm/normalized_op_sizes[i]).cpu().detach().numpy(),"o")
    else:
        for i, op_norm in enumerate(op_norms):
            plt.semilogy(i, op_norm.cpu().detach().numpy(),"o")

    plt.xticks(np.arange(num_ops), op_names, rotation=90)
    plt.xlim(-1, num_ops)    
    plt.ylim(1e-6, 1e2)    
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def acc_n_loss(train_loss, test_acc, filename, train_acc=None, test_loss=None):
    if train_acc is not None and test_loss is not None:
        fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.2))
        fig.suptitle('Loss and Acc')
        axs[0,0].semilogy(train_loss)
        axs[0,0].grid(True)
        axs[0,0].set_xlabel('Epochs')
        axs[0,0].set_ylabel('Training loss')

        axs[0,1].plot(train_acc)
        axs[0,1].grid(True)
        axs[0,1].set_ylim(0,101)
        axs[0,1].set_yticks(np.arange(0, 101, 5))
        axs[0,1].set_xlabel('Epochs')
        axs[0,1].set_ylabel('Train accuracy (in %)')

        axs[1,0].semilogy(test_loss)
        axs[1,0].grid(True)
        axs[1,0].set_xlabel('Epochs')
        axs[1,0].set_ylabel('Test loss')

        axs[1,1].plot(test_acc)
        axs[1,1].grid(True)
        axs[1,1].set_ylim(0,101)
        axs[1,1].set_yticks(np.arange(0, 101, 5))
        axs[1,1].set_xlabel('Epochs')
        axs[1,1].set_ylabel('Test accuracy (in %)')

        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)       
        
    elif train_acc is not None and test_loss is None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 4.8))
        fig.suptitle('Loss and Acc')
        ax1.semilogy(train_loss)
        ax1.grid(True)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training loss')

        ax2.plot(train_acc)
        ax2.grid(True)
        ax2.set_ylim(0,101)
        ax2.set_yticks(np.arange(0, 101, 5))
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Train accuracy (in %)')

        ax3.plot(test_acc)
        ax3.grid(True)
        ax3.set_ylim(0,101)
        ax3.set_yticks(np.arange(0, 101, 5))
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Test accuracy (in %)')

        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        
    elif train_acc is None and test_loss is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 4.8))
        fig.suptitle('Loss and Acc')
        ax1.semilogy(train_loss)
        ax2.plot(test_acc)
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

        
def group_model_params_by_cell(model, network, mu=None):
  """
  This functions put the same operation of different cells into the same vector (the group will be passed to optimizer).
    
  One operation may consist of several suboperations. For example, op 3 of edge 7 consists of (_ops.7._ops.3.op.1.weight, _ops.7._ops.3.op.2.weight, _ops.7._ops.3.op.5.weight, _ops.7._ops.3.op.6.weight). Op 3 of edge 7 from all cells of the same type (normal or reduce) will be grouped as one, "_ops.7._ops.3". During discretization, the operation "_ops.7._ops.3" will be pruned if its norm is smaller than the pruning threshold.

  The operations are treated differently in "group_model_params_by_stage"! In "group_model_params_by_stage", all suboperations of an operation are treated independently. During discretization, the operation will be pruned if one of its suboperations is smaller than the pruning threshold. For example, the operation "_ops.7._ops.3" will be pruned if any of its suboperations has a norm that is smaller than the pruning threshold (recall that the suboperations are linearly placed. One of them being 0 means the entire operation is 0.)
  """
    
  assert network.num_ops <=9, "The number of operations should be smaller than 10 (but got {}).".format(network.num_ops)
  assert network.num_edges <=100, "The number of edges should be smaller than 100 (but got {}).".format(network.num_edges)
  
  """
  The operations that are trainable but not prunable should be separated from trainable and prunable operations. "Unprunable" means these are the operations that will be definitely kept in the final network (such as the preprocessing layer, the final classifier layer, and the preprocessing of input nodes in each cell), in contrast to the operations that may be pruned away after searching is completed.
  """    
  ops_unprunable = []
  for op in model.stem:
        for param in op.parameters():
            ops_unprunable.append(param)
                
  """model.global_pooling is before classifier, but it is not trainable"""
  classifier_weight, classifier_bias = model.classifier.parameters()
  ops_unprunable.extend([classifier_weight, classifier_bias])
  
  """The operations that are prunable are put in a separate dictionary."""
  ops_prunable_normal = dict()
  ops_prunable_reduce = dict()
  for edge in range(network.num_edges):
    for op in range(network.num_ops):
        ops_prunable_normal["_ops.{}._ops.{}".format(edge, op)] = []
        ops_prunable_reduce["_ops.{}._ops.{}".format(edge, op)] = []
        
  for cell_index, m in enumerate(model.cells):
        op_index = 0
        edge_index = 0
        
        for name, param in m.named_parameters():
            """
            An example of "name" is _ops.8._ops.4.op.2.weight, where 8 represents the edge, 4 is the op index, and 2 is the subop of op 4 (op 4 consists of several subops).
            """
            if "_ops" in name:
                if "_ops.0._ops.0" in name: # beginning of a new cell
                    cur_op_name = name[0:13] # assuming the number of cells < 10
                    pre_op_name = cur_op_name
                else:
                    if edge_index <= 9:
                        cur_op_name = name[0:13] #example: extract "_ops.3._ops.4" from "_ops.3._ops.4.op.2.weight"
                    else:
                        cur_op_name = name[0:14] #example: extract "_ops.13._ops.4" from "_ops.13._ops.4.op.2.weight"
                
                if cur_op_name == pre_op_name: #still the same op
                    pass
                else: # current op is a new op
                    op_index += 1
                    if op_index == network.num_ops: # the current op belongs to a new edge
                        op_index = 0
                        edge_index += 1
                    else: # still the same edge
                        pre_op_name = cur_op_name
                        
                    if edge_index <= 9: # get the name of the current (new) op
                        cur_op_name = name[0:13]
                    else:
                        cur_op_name = name[0:14]
                           
#                 print("  name is      {}, edge index is {}, op index is {}".format(name, edge_index, op_index))
#                 print("  cur_op_name: {}".format(cur_op_name))
#                 print("  pre_op_name: {}".format(pre_op_name))
                
                if cell_index in network.reduce_cell_indices:
                    ops_prunable_reduce[cur_op_name].append(param)
                else:
                    ops_prunable_normal[cur_op_name].append(param)
            else:
                ops_unprunable.append(param)

                
  """
  define the parameter groups that will be passed to the optimizer. Prunable operations will have a nonzero weight decay (mu), while nonprunable operations do not have a mu.
  """
  model_params = []
  for op_name, op_param in ops_prunable_normal.items():
    model_params.append(dict(params=op_param, label="normal", op_name=op_name, weight_decay=mu))    
  for op_name, op_param in ops_prunable_reduce.items():
    model_params.append(dict(params=op_param, label="reduce", op_name=op_name, weight_decay=mu))  
  model_params.append(dict(params=ops_unprunable, label="unprunable", op_name="unprunable", weight_decay=None))
  
  return model_params

def group_model_params_by_stage(model, network, mu=None):
  """
  This functions put the same operation in different cells in the same stage into the same group (Example: stage normal 1, stage reduce 1, stage normal 2, stage reduce 2, stage normal 3. Each stage consists of several cells. Here normal cells and reduce cells are not differentiated.)
  
  The operations are treated differently in "group_model_params_by_cell" and "group_model_params_by_stage"!
  
  "group_model_params_by_stage": All suboperations of an operation are treated independently. For example, op 3 of edge 7 has the following suboperations: _ops.7._ops.3.op.1.weight, _ops.7._ops.3.op.2.weight, _ops.7._ops.3.op.5.weight, _ops.7._ops.3.op.6.weight. During discretization, the operation will be pruned if one of its suboperations is smaller than the pruning threshold (recall that the suboperations are linearly placed. One of them being 0 means the entire operation is 0.).
  
  In "group_model_params_by_cell", all suboperations of an operation are grouped into a single vector. For example, op 3 of edge 7 has the following suboperations: _ops.7._ops.3.op.1.weight, _ops.7._ops.3.op.2.weight, _ops.7._ops.3.op.5.weight, _ops.7._ops.3.op.6.weight. All of these suboperations are grouped into a single vector called "_ops.7._ops.3". During discretization, the operation "_ops.7._ops.3" will be pruned if its norm is smaller than the pruning threshold.
  """

  """
  The operations that are trainable but not prunable should be separated from trainable and prunable operations. "Unprunable" means these are the operations that will be definitely kept in the final network (such as the preprocessing layer, the final classifier layer, and the preprocessing of input nodes in each cell), in contrast to the operations that may be pruned away after searching is completed.
  """
  ops_unprunable = []
  for op in model.stem:
        for param in op.parameters():
            ops_unprunable.append(param)
                
  """model.global_pooling is before classifier, but it is not trainable"""
  classifier_weight, classifier_bias = model.classifier.parameters()
  ops_unprunable.extend([classifier_weight, classifier_bias])


  """The operations that are prunable are put in a separate dictionary."""
  ops_prunable = {}
  stage_normal_index = 1
  stage_reduce_index = 1
  for cell_index, m in enumerate(model.cells):
        for name, param in m.named_parameters():
            if "_ops" in name: # ops of the intermediate nodes between input nodes and output node
                if cell_index in network.reduce_cell_indices: #reduce cell
                    op_name = "stage_reduce_{}.{}".format(stage_reduce_index, name)
                else: #normal cell
                    op_name = "stage_normal_{}.{}".format(stage_normal_index, name)

                if op_name in ops_prunable:
                    ops_prunable[op_name].append(param)
                else:
                    ops_prunable[op_name] = [param]
            else: # preprocessing ops of the input nodes
                ops_unprunable.append(param)
                        
        if cell_index in network.reduce_cell_indices:
            stage_reduce_index += 1
            stage_normal_index += 1

            
  """
  define the parameter groups that will be passed to the optimizer. Prunable operations will have a nonzero weight decay (mu), while nonprunable operations do not have a mu.
  """
  model_params = []
  for op_name, op_param in ops_prunable.items():
    model_params.append(dict(params=op_param, label="prunable", op_name=op_name, weight_decay=mu))    
  model_params.append(dict(params=ops_unprunable, label="unprunable", op_name="unprunable", weight_decay=None))
  
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
        
    num_ops = len(op_norm_normal_dict)
    f1 = plt.figure(num=None, figsize=(num_ops*0.15, 6), dpi=100, facecolor='w', edgecolor='k')    
    op_names = []
    for i, (op_name, op_norm) in enumerate(op_norm_normal_dict.items()):
        op_names.append(op_name)
        plt.semilogy(i, op_norm.cpu().detach().numpy(), "o")
        
    plt.xticks(np.arange(num_ops), op_names, rotation=90)
    plt.xlim(-1, num_ops)
    plt.ylim(1e-6, 1e2)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("{}_normal.png".format(filename))
    plt.close()
    
    
    num_ops = len(op_norm_reduce_dict)
    f2 = plt.figure(num=None, figsize=(num_ops*0.15, 6), dpi=100, facecolor='w', edgecolor='k')    
    op_names = []
    for i, (op_name, op_norm) in enumerate(op_norm_reduce_dict.items()):
        op_names.append(op_name)
        plt.semilogy(i, op_norm.cpu().detach().numpy(), "o")
        
    plt.xticks(np.arange(num_ops), op_names, rotation=90)
    plt.xlim(-1, num_ops)
    plt.ylim(1e-6, 1e2)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("{}_reduce.png".format(filename))
    plt.close()
    
def compute_op_norm_across_stages(model_params):
    """
    compute the norm of the vector containing the weights of the same operation in all cells in each stage stages (e.g., sep_conv_3x3 in normal_stage_1, reduce_stage_1, normal_stage 2, reduce_stage_2, normal_stage_3)
    """
    op_norm_dict = {}
    for operation in model_params:
        if operation["label"] == "unprunable":
            continue
        
        params = operation["params"]
        params_norm_square = 0
        for param in params:
            params_norm_square += torch.norm(param) ** 2
         
        op_norm_dict[operation["op_name"]] = torch.sqrt(params_norm_square)
    
    return op_norm_dict

def plot_op_norm_across_stages(model_params, filename):
    op_norm_dict = compute_op_norm_across_stages(model_params)
        
    num_ops = len(op_norm_dict)
    f = plt.figure(num=None, figsize=(num_ops*0.15, 6), dpi=100, facecolor='w', edgecolor='k')
    
    op_names = []
    for i, (op_name, op_norm) in enumerate(op_norm_dict.items()):
        op_names.append(op_name)
        plt.semilogy(i, op_norm.cpu().detach().numpy(), "o")
    
    plt.xticks(np.arange(num_ops), op_names, rotation=90)
    plt.xlim(-1, num_ops)
    plt.ylim(1e-6, 1e2)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def discretize_search_model_by_cell(model_path, network_eval, network_search, threshold, CIFAR_CLASSES = 10):
  """
  remove the ops with a small norm and the discrete cell will be scaled up for evaluation
  
  All suboperations of an operation are grouped into a single vector. For example, op 3 of edge 7 has the following suboperations: _ops.7._ops.3.op.1.weight, _ops.7._ops.3.op.2.weight, _ops.7._ops.3.op.5.weight, _ops.7._ops.3.op.6.weight. All of these suboperations are grouped into a single vector called "_ops.7._ops.3". During discretization, the operation "_ops.7._ops.3" will be pruned if its norm is smaller than the pruning threshold.
  """
  model = SearchNetwork(network_search.init_channels, CIFAR_CLASSES, network_search.cells, network_search.criterion)
  model = model.cuda()
  model.load_state_dict(torch.load(model_path))

  model_params = group_model_params_by_cell(model, network_search)
  op_norm_normal, op_norm_reduce = compute_op_norm_across_cells(model_params)

  alpha_normal = []
  alpha_edge = []
  edge_index = 0  
  for op_index, (op_name, op_norm) in enumerate(op_norm_normal.items()): # iterate over the operations (not suboperations)
        if  edge_index * network_search.num_ops <= op_index < (edge_index + 1) * network_search.num_ops:
            if op_norm <= threshold:
                alpha_edge.append(0)
            else:
                alpha_edge.append(1)
            if op_index == (edge_index + 1) * network_search.num_ops - 1:
                alpha_normal.append(alpha_edge)
                alpha_edge = []
                edge_index += 1
  alpha_normal = torch.tensor(alpha_normal)

  alpha_reduce = []
  alpha_edge = []
  edge_index = 0
  for op_index, (op_name, op_norm) in enumerate(op_norm_reduce.items()):        
        if  edge_index * network_search.num_ops <= op_index < (edge_index + 1) * network_search.num_ops:
            if op_norm <= threshold:
                alpha_edge.append(0)
            else:
                alpha_edge.append(1)
            if op_index == (edge_index + 1) * network_search.num_ops - 1:
                alpha_reduce.append(alpha_edge)
                alpha_edge = []
                edge_index += 1
  alpha_reduce = torch.tensor(alpha_reduce)

  alpha_network = []
  num_reduce_cell = len(network_eval.reduce_cell_indices)
  cur_reduce_cell = 0
  for cell_index in range(network_eval.cells): # cells up to the last reduce cell (included)
    if cell_index < network_eval.reduce_cell_indices[cur_reduce_cell]:
        alpha_network.append((False, np.vstack(alpha_normal)))
    elif cell_index == network_eval.reduce_cell_indices[cur_reduce_cell]:
        alpha_network.append((True,  np.vstack(alpha_reduce)))
        cur_reduce_cell += 1        
        if cur_reduce_cell == num_reduce_cell:
            break            
  # cells after the last reduce cell            
  for cell_index in range(network_eval.reduce_cell_indices[-1]+1, network_eval.cells):
        alpha_network.append((False, np.vstack(alpha_normal)))
        
  genotype_network = get_genotype(model.genotype(), alpha_network)

  return alpha_network, genotype_network


def discretize_search_model_by_stage(model_path, network_eval, network_search, threshold=1e-5, CIFAR_CLASSES = 10, swap_stage=False):
  """
  remove the operations with a small norm and the discrete cell will be scaled up for evaluation
  
  All suboperations of an operation are grouped into a single vector. For example, op 3 of edge 7 has the following suboperations: _ops.7._ops.3.op.1.weight, _ops.7._ops.3.op.2.weight, _ops.7._ops.3.op.5.weight, _ops.7._ops.3.op.6.weight. All of these suboperations are grouped into a single vector called "_ops.7._ops.3". During discretization, the operation "_ops.7._ops.3" will be pruned if its norm is smaller than the pruning threshold.
  """
  assert network_search.num_ops <=9, "The number of operations should be smaller than 10 (but got {}).".format(network.num_ops)
  assert network_search.num_edges <=100, "The number of edges should be smaller than 100 (but got {}).".format(network.num_edges)

  model = SearchNetwork(network_search.init_channels, CIFAR_CLASSES, network_search.cells, network_search.criterion)
  model = model.cuda()
  model.load_state_dict(torch.load(model_path))

  model_params = group_model_params_by_stage(model, network_search)
  op_norm_dict = compute_op_norm_across_stages(model_params)

  stage_name = "stage_normal_1"
  op_index = 0
  edge_index = 0
  alpha_edge = [1]
  alpha_stage = []
  alpha_all_stages = []
  for op_name, op_norm in op_norm_dict.items(): # iterate over the suboperations (not the operations as in cell discretization)
        if stage_name == op_name[0:14]: # still in the same stage
            pass
        else: # in a new stage
            stage_name = op_name[0:14]
            alpha_stage.append(alpha_edge)
            alpha_all_stages.append(alpha_stage)
            alpha_stage = []
            op_index = 0
            edge_index = 0
            alpha_edge = [1]
        
        if edge_index == int(op_name[20: 21 + (edge_index>=10)]): # still in the same edge
            pass
        else: # in a new edge
            alpha_stage.append(alpha_edge)
            op_index = 0
            edge_index += 1
            alpha_edge = [1]
            
        if op_index == int(op_name[27 + (edge_index>=10)]): # still in the same op (one operation may consist of several suboperations)
            # the operation will be kept if ALL of its suboperations have a norm larger than the threshold
            alpha_edge[-1] *= (op_norm.cpu().detach().numpy() > threshold)
        else: # in a new op
            alpha_edge.append(int(op_norm.cpu().detach().numpy() > threshold))
            op_index += 1  
  alpha_stage.append(alpha_edge)
  alpha_all_stages.append(alpha_stage)

    
  if not swap_stage:
      """
      normal scaling (in contrast to the swapped scaling below):
      [BEGIN] stage_normal_1, stage_reduce_1, stage_normal_2, stage_reduce_2, stage_normal_3 [END]
      
      A more flexible stage is currently not supported, for example:
      [BEGIN] stage_normal_1, stage_reduce_1, stage_normal_2, stage_reduce_2, stage_normal_3, stage_normal_4 [END]
      """
      alpha_network = []
      num_reduce_cell = len(network_eval.reduce_cell_indices)
      cur_reduce_cell = 0
      for cell_index in range(network_eval.cells):
        if cell_index < network_eval.reduce_cell_indices[cur_reduce_cell]:
            alpha_network.append((False, np.vstack(alpha_all_stages[2*cur_reduce_cell])))
        elif cell_index == network_eval.reduce_cell_indices[cur_reduce_cell]:
            alpha_network.append((True, np.vstack(alpha_all_stages[2*cur_reduce_cell+1])))
            cur_reduce_cell += 1        
            if cur_reduce_cell == num_reduce_cell:
                break            
      for cell_index in range(network_eval.reduce_cell_indices[-1]+1, network_eval.cells):
            alpha_network.append((False, np.vstack(alpha_all_stages[2*cur_reduce_cell])))

  elif swap_stage:
      """
      swap stages when scaling, to test the importance of the 'correct ordering' of stages found in search.
      The swapped stages look like this:
      [BEGIN] stage_normal_3, stage_reduce_2, stage_normal_2, stage_reduce_1, stage_normal_1 [END]      
      """
      alpha_network = []
      last_stage_index = len(alpha_all_stages) - 1
      num_reduce_cell = len(network_eval.reduce_cell_indices)
      cur_reduce_cell = 0
      for cell_index in range(network_eval.cells):
        if cell_index < network_eval.reduce_cell_indices[cur_reduce_cell]:
            alpha_network.append((False, np.vstack(alpha_all_stages[last_stage_index - 2*cur_reduce_cell])))
        elif cell_index == network_eval.reduce_cell_indices[cur_reduce_cell]:
            alpha_network.append((True, np.vstack(alpha_all_stages[last_stage_index - (2*cur_reduce_cell+1)])))
            cur_reduce_cell += 1        
            if cur_reduce_cell == num_reduce_cell:
                break            
      for cell_index in range(network_eval.reduce_cell_indices[-1]+1, network_eval.cells):
            alpha_network.append((False, np.vstack(alpha_all_stages[0])))
    
  genotype_network = get_genotype(model.genotype(), alpha_network)

  return alpha_network, genotype_network

def get_genotype(genotype_supernet, alpha_network):
    genotype_network = []
    for i, (reduce_cell, alpha_cell) in enumerate(alpha_network):
        alpha_cell = alpha_cell.flatten()
        indices = np.where(alpha_cell == 1)[0]
        if reduce_cell:
            genotype_network.append([genotype_supernet.reduce[x] for x in indices.astype(int)])
        else:
            genotype_network.append([genotype_supernet.normal[x] for x in indices.astype(int)])
    return genotype_network

def visualize_cell(alpha, network, filename):
  g = Digraph(
      format="pdf",
      edge_attr=dict(fontsize='10', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  # input node
  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  
  steps = network.steps

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  step_offset = 0
  active_nodes = []
  for ii in range(2, steps + 2):
    for jj in range(step_offset, step_offset + ii):
      if sum(alpha[jj]) == 0:
        continue
      
      if ii-2 not in active_nodes:
        active_nodes.append(ii-2)
      for op_index, active_op in enumerate(alpha[jj]):
        if active_op:
            op = network.ops[op_index]
            j = jj - step_offset
            
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)
            v = str(ii - 2)
            g.edge(u, v, label=op, fillcolor="gray")
    step_offset += ii

  """output node"""
  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in active_nodes:
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=False)
  os.remove(filename)

def visualize_cell_in_network(network, alpha_network, scale_type, folder_path):
  if scale_type == "cell":
    visualize_cell(alpha_network[network.reduce_cell_indices[0]-1][1], network, "{}/cell_normal".format(folder_path))
    visualize_cell(alpha_network[network.reduce_cell_indices[0]][1], network, "{}/cell_reduce".format(folder_path))
  elif scale_type == "stage":
    stage_index = 1
    for cell in network.reduce_cell_indices:
        visualize_cell(alpha_network[cell-1][1], network, "{}/cell_in_stage_normal_{}".format(folder_path, stage_index))
        visualize_cell(alpha_network[cell][1], network, "{}/cell_in_stage_reduce_{}".format(folder_path, stage_index))
        stage_index += 1
    visualize_cell(alpha_network[cell+1][1], network,  "{}/cell_in_stage_normal_{}".format(folder_path, stage_index))