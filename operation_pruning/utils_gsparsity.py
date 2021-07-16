import os
import torch
import torch.nn as nn
import numpy as np
from graphviz import Digraph

def visualize_cell(alpha, network, filename):
  colors = ["sienna3", "red", "green4", "royalblue", "magenta"]
  num_colors = len(colors)
  g = Digraph(
      format="pdf",
      edge_attr=dict(fontsize='10'),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2'),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  # input node
  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  
  steps = network.steps

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  edge_offset = 0
  active_nodes = []
  color_index = 0
  for step_index in range(2, steps + 2):
    for edge_index in range(edge_offset, edge_offset + step_index):
      if sum(alpha[edge_index]) == 0:
        continue
      
      if step_index-2 not in active_nodes:
        active_nodes.append(step_index-2)
      for op_index, active_op in enumerate(alpha[edge_index]):
        if active_op:
            op = network.ops[op_index]
            j = edge_index - edge_offset
            
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)
            v = str(step_index - 2)
            color = colors[color_index]
            g.edge(u, v, label=op, color=color, fillcolor=color, fontcolor=color)
            color_index = (color_index+1) % num_colors
    edge_offset += step_index

  """output node"""
  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in active_nodes:
    g.edge(str(i), "c_{k}", fillcolor="gray", penwidth="2.0")

  g.render(filename, view=False)
  os.remove(filename)

    
def discretize_model_by_operation(model, network_eval, genotype, threshold, folder_path, num_edges = 8):       
    edge_offset = [0] #edge_offset = [0, 2, 5, 9]
    for i in range(2, network_eval.steps + 2 - 1):
        edge_offset.append(edge_offset[-1] + i)
        
    alpha_cell_list = []
    alpha_network = []
    cell_inactive = []
    for cell_index, m in enumerate(model.cells):
        assert cell_index <= network_eval.cells - 1, "The number of cells in the loaded model is different from the number of cells expected ({}).".format(network_eval.cells)
        
        alpha_cell = np.zeros((network_eval.num_edges, network_eval.num_ops))
        if cell_index in network_eval.reduce_cell_indices:
            reduce_cell = True
            op_names, indices = zip(*genotype.reduce)
        else:
            reduce_cell = False
            op_names, indices = zip(*genotype.normal)

        for edge_index, (op_name, index) in enumerate(zip(op_names, indices)):
            op_index = network_eval.ops.index(op_name)
            node_index = edge_index // 2
            alpha_cell[edge_offset[node_index] + index][op_index] = 1
            
        for name, param in m.named_parameters():
            if "_ops" in name:
                if "bias" in name:
                    continue
                edge_index = int(name[5])
                node_index = edge_index // 2
                op_name = op_names[edge_index]
                index = indices[edge_index]
                op_index = network_eval.ops.index(op_name)
                alpha_cell[edge_offset[node_index] + index][op_index] *= (torch.norm(param) > threshold)
        alpha_cell_list.append(alpha_cell)
        cell_inactive.append(np.sum(alpha_cell) == 0)
        
#         print("cell {}, alpha_cell {}".format(cell_index, alpha_cell))
        
    assert cell_index == network_eval.cells - 1, "The number of cells in the loaded model is different from the number of cells expected ({}).".format(network_eval.cells)

    """detecting redundant edges..."""
#     print("detecting redundant edges...")
    
    genotype_network = []
    for cell_index in range(0, network_eval.cells):
        
        alpha_cell = alpha_cell_list[cell_index]
        if cell_index == 0:
            node_inactive_list = [False, False]
        elif cell_index == 1:
            node_inactive_list = [False, cell_inactive[cell_index - 1]]
        else:
            node_inactive_list = [cell_inactive[cell_index - 2], cell_inactive[cell_index - 1]]
        for node_index in range(2, network_eval.steps + 2):
            for edge_index in range(edge_offset[node_index-2], edge_offset[node_index-2] + node_index):
#                 print("    edge_index {}".format(edge_index))
                if node_inactive_list[edge_index - edge_offset[node_index-2]]:
                    for op_index in range(network_eval.num_ops):
                        alpha_cell[edge_index][op_index] = 0
               
            num_active_ops = 0
            for edge_index in range(edge_offset[node_index-2], edge_offset[node_index-2] + node_index):
                for op_index in range(network_eval.num_ops):
                    num_active_ops += alpha_cell[edge_index][op_index]
                        
            step_inactive = (num_active_ops == 0)
#             print("node_index {}, num_active_ops: {}".format(node_index - 2, num_active_ops))
            
            node_inactive_list.append(step_inactive)
        cell_inactive[cell_index] = (sum(node_inactive_list[2:]) == network_eval.steps)
        alpha_network.append((cell_index in network_eval.reduce_cell_indices, alpha_cell))
                
#         print("cell {}, alpha_cell {}".format(cell_index, alpha_cell))
#         print("inactive node list: {}".format(node_inactive_list))
#         print("cell inactive? {}".format(cell_inactive[cell_index]))
        
        visualize_cell(alpha_cell, network_eval, "{}/cell_{:02d}".format(folder_path, cell_index))

        genotype_cell = []
        for node_index in range(network_eval.steps):
            for edge_index in range(edge_offset[node_index], edge_offset[node_index] + node_index + 2):
                for kk in range(network_eval.num_ops):
                    if alpha_cell[edge_index][kk] == 1:
                        op_name = network_eval.ops[kk]
                        source_node = edge_index - edge_offset[node_index]
                        genotype_cell.append((op_name, source_node))
#         print(genotype_cell)
    
        genotype_network.append(genotype_cell)
        
    return alpha_network, genotype_network