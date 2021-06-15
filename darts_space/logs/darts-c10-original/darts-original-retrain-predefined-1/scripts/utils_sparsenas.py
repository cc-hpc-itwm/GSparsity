import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model_search import Network as SearchNetwork

def plot_individual_op_norm(model,filename, figsize_width=150, normalize=False):
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


def acc_n_loss(train_loss, test_top1, filename, train_top1=None):
    if train_top1 is None:
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
    else:
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

def group_model_params_by_cell(model, mu=None, reduce_cell_indices=[2, 5], num_edges=14, num_ops=7):
    #This functions put the same operation of different cells into the same group (the group will be passed to optimizer)
  param_dict_normal = dict()
  param_dict_reduce = dict()
  param_others = []

  for edge in range(num_edges):
    for op in range(num_ops):
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
                    if op_index == num_ops:
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
                
                if cell_index in reduce_cell_indices:
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

def group_model_params_by_stage(model, mu, reduce_cell_indices = [2, 5], num_edges=14, num_ops=7):
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
                if cell_index in reduce_cell_indices: #reduction cell
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
                        
        if cell_index in reduce_cell_indices:
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
    
    
def plot_op_norm_across_stages(model_params, filename):
    op_norm_dict = {}
    for operation in model_params:
        if operation["label"] == "unprunable":
            continue
        
        params = operation["params"]
        params_norm_square = 0
        for param in params:
            params_norm_square += torch.norm(param) ** 2
         
        op_norm_dict[operation["name"]] = torch.sqrt(params_norm_square)
        
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
    all_cells.append((cell.reduction,np.vstack(cell_struct)))
  #print(all_cells)
  return all_cells

def discretize_search_model(filename, init_channels=16, layers=8, threshold=1e-5, CIFAR_CLASSES = 10):
# removing the ops with a small norm and the discrete cell will be scaled up for evaluation
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = SearchNetwork(init_channels, CIFAR_CLASSES, layers, criterion)
  model = model.cuda()
  model.load_state_dict(torch.load(filename))
  num_edges = 14
  num_ops = 7

  model_params = group_model_params_by_cell(model, reduce_cell_indices=[2, 5], num_edges=num_edges, num_ops=num_ops)
  op_norm_normal, op_norm_reduce = compute_op_norm_across_cells(model_params)

  alpha_normal = []
  alpha = []
  edge_index = 0  
  for index, (op_name, op_norm) in enumerate(op_norm_normal.items()):        
        if  edge_index * num_ops <= index < (edge_index + 1) * num_ops:            
#             print("index {} (edge {}): op {}, norm {}".format(index, edge_index, op_name, op_norm))
            if op_norm <= threshold:
                alpha.append(0)
            else:
                alpha.append(1)
            if index == (edge_index + 1) * num_ops - 1:
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
        if  edge_index * num_ops <= index < (edge_index + 1) * num_ops:            
#             print("index {} (edge {}): op {}, norm {}".format(index, edge_index, op_name, op_norm))
            if op_norm <= threshold:
                alpha.append(0)
            else:
                alpha.append(1)
            if index == (edge_index + 1) * num_ops - 1:
#                 print("edge {}: alpha {}".format(edge_index, alpha))
                alpha_reduce.append(alpha)
                alpha = []
                edge_index += 1
  alpha_reduce = torch.tensor(alpha_reduce)
#   print("alpha_reduce: {}".format(alpha_reduce))

  alphas = (alpha_normal, alpha_reduce)
#   print("joint tuple: {}".format(alphas))
  
  return alphas