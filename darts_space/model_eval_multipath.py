import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path
import numpy as np

class MixedOp(nn.Module): 
#operation at an edge: an edge may consist of several operations
  def __init__(self, op_list):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    if op_list:
        self._active = True
    else:
        self._active = False
    for op in op_list:
      self._ops.append(op)

  def forward(self, x):
    return sum(op(x) for op in self._ops)


class Cell(nn.Module):
    def __init__(self, steps, genotype_cell, alpha_cell, C_prev_prev, C_prev, C, reduce, reduce_prev, concat, primitives_cell):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)
        self.op_list = primitives_cell
        self._steps = steps
        self._concat = concat
        self.multiplier = len(concat)
    
        if C_prev_prev == 0: # cell k-2 does NOT have any active operations
            self.preprocess0 = None
        else:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C) if reduce_prev else ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        
        if C_prev == 0: # cell k-1 does NOT have any active operations
            self.preprocess1 = None
        else:
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if genotype_cell:
            _, indices = zip(*genotype_cell)
        else:  # the current cell k does NOT have any active operations
            indices = []
        
        self._compile(C, indices, reduce, alpha_cell) 


    def _compile(self, C, indices, reduce, alpha_cell):
        self._ops = nn.ModuleList()
        index = 0
        for edge_index, alpha_edge in enumerate(alpha_cell):
            edge_active_ops = []
            primitives_edge = self.op_list[edge_index]
            for op_idx, val in enumerate(alpha_edge):
                if val:
                    # In reduce cells, all operations adjacent to input nodes are of stride 2.
                    stride = 2 if reduce and indices[index] < 2 else 1
                    op = OPS[primitives_edge[op_idx]](C, stride, True)
                    edge_active_ops.append(op)

            edge_op = MixedOp(edge_active_ops)
            self._ops.append(edge_op)
            index += sum(alpha_edge)


    def forward(self, s0, s1, drop_prob):
        if self._concat == []:
            return torch.zeros(1)
    
        s0 = self.preprocess0(s0) if self.preprocess0 is not None else torch.zeros(1)
        s1 = self.preprocess1(s1) if self.preprocess1 is not None else torch.zeros(1)
        node_outputs = [s0, s1]

        edge_offset = 0
        for cur_node in range(self._steps): #iterate through the intermediate nodes between input nodes and output node
            edge_output_list = []
            for pre_node, pre_node_output in enumerate(node_outputs):
                edge_op = self._ops[edge_offset+pre_node]
                if edge_op._active:
                    edge_output = edge_op(pre_node_output)
                    if self.training and drop_prob > 0.:
                        if not isinstance(edge_op, Identity):                    
                            edge_output = drop_path(edge_output, drop_prob)
                    edge_output_list.append(edge_output)
            cur_node_output = sum(edge_output_list)
            edge_offset += len(node_outputs)
            node_outputs.append(cur_node_output)
        cell_output = torch.cat([node_outputs[i] for i in self._concat], dim=1)
        return cell_output


class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class NetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, cells, auxiliary, genotype_network, alpha_network, reduce_cell_indices, steps, primitives):
        """
        build the network from alpha_network
        in the supernet, all ops are active
        """
        super(NetworkCIFAR, self).__init__()
        self._cells = cells
        self._auxiliary = auxiliary
        self._steps = steps
        self._reduce_cell_indices = reduce_cell_indices

        stem_multiplier = 3
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(C_curr))
    
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduce_prev = False
        for i in range(cells):
            reduce_cell, alpha_cell = alpha_network[i]
            genotype_cell = genotype_network[i]
        
            if genotype_cell == []:
                print("Cell {} does NOT have any active operations.".format(i))
      
            if i in self._reduce_cell_indices:
                assert reduce_cell == True, "This cell is expected to be a reduce cell but it is not!"
                C_curr *= 2
                reduce = True
                primitives_cell = primitives['primitives_reduct']
            else:
                assert reduce_cell == False, "This cell is expected to be a normal cell but it is not!"
                reduce = False
                primitives_cell = primitives['primitives_normal']
        
            concat = set()
            edge_offset = 0
            op_offset = 0
            alpha_cell_new = []
            for step_index in range(2, self._steps + 2):
                """
                Iterate through the intermediate nodes between input nodes and output node. The output of active nodes (with at least one active edge) will be concatenated to form the cell output.
                """
                for edge_index in range(edge_offset, edge_offset + step_index):
                    primitives_edge = primitives_cell[edge_index]
                    alpha_edge = alpha_cell[op_offset:op_offset+len(primitives_edge)]
                    alpha_cell_new.append(alpha_edge)
                    op_offset += len(primitives_edge)
                    if sum(alpha_edge) > 0:
                        concat.add(step_index)
                edge_offset += step_index
            cell = Cell(self._steps, genotype_cell, alpha_cell_new, C_prev_prev, C_prev, C_curr, reduce, reduce_prev, concat, primitives_cell)
            self.cells += [cell]
    
            reduce_prev = reduce
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
            if i == self._reduce_cell_indices[-1]:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)


    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == self._reduce_cell_indices[-1]:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, cells, auxiliary, genotype_network, alpha_network, reduce_cell_indices, steps, primitives):
    
    super(NetworkImageNet, self).__init__()
    self._cells = cells
    self._auxiliary = auxiliary
    self._steps = steps
    self._reduce_cell_indices = reduce_cell_indices

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduce_prev = True
    for i in range(cells):
      alpha_cell = alpha_network[i]
      genotype_cell = genotype_network[i]

      if genotype_cell == []:
        print("Cell {} does NOT have any active operations.".format(i))

      if i in self._reduce_cell_indices: #alpha_cell[0] is True (if it's a reduce cell) or False (if it's a normal cell)
        assert alpha_cell[0] == True, "This cell is expected to be a reduce cell but it is not!"
        C_curr *= 2
        reduce = True
        primitives_cell = primitives['primitives_reduct']
      else:
        assert alpha_cell[0] == False, "This cell is expected to be a normal cell but it is not!"
        reduce = False
        primitives_cell = primitives['primitives_normal']
        
      concat = []
      edge_offset = 0
      for step_index in range(2, self._steps + 2):
        """
        Iterate through the intermediate nodes between input nodes and output node. The output of active nodes (with at least one active edge) will be concatenated to form the cell output.
        """
        for edge_index in range(edge_offset, edge_offset + step_index):
            if sum(alpha_cell[1][edge_index]) > 0:
                concat.append(step_index)
                break
        edge_offset += step_index

      cell = Cell(self._steps, genotype_cell, alpha_cell[1], C_prev_prev, C_prev, C_curr, reduce, reduce_prev, concat, primitives_cell)
      reduce_prev = reduce
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * cells // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    
    self.global_pooling = nn.AvgPool2d(7)
    
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, drop_prob=self.drop_path_prob)
      if i == self._reduce_cell_indices[-1]:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux