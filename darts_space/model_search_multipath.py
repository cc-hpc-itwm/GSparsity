import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import Genotype


class MixedOp(nn.Module):
    def __init__(self, C, stride, primitives):
        super(MixedOp, self).__init__()
        self._op = nn.ModuleList()
        for primitive in primitives:
            op = OPS[primitive](C, stride, False)
            self._op.append(op)

    def forward(self, x):
        return sum(op(x) for op in self._op)


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduce, reduce_prev, primitives_cell):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)
        self.reduce = reduce

        if reduce_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps # number of intermediate nodes
        self._multiplier = multiplier

        self._edge = nn.ModuleList()
        self._bns = nn.ModuleList()

        edge_index = 0
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduce and j < 2 else 1 # the first two input nodes have a stride of 2
                op = MixedOp(C, stride, primitives_cell[edge_index])
                self._edge.append(op)
                edge_index += 1

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._edge[offset+j](h) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, cells, criterion, primitives, steps=4,
               multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._cells = cells
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    
    self.primitives = primitives

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduce_prev = False
    for i in range(cells):
      if i in [cells//3, 2*cells//3]:
        C_curr *= 2
        reduce = True
        primitives_cell = primitives['primitives_reduct']
      else:
        reduce = False
        primitives_cell = primitives['primitives_normal']
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduce, reduce_prev, primitives_cell)
      reduce_prev = reduce
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def genotype(self):

    def _parse(normal=True):
      PRIMITIVES = self.primitives['primitives_normal' if normal else
                                   'primitives_reduct']
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        edges = range(i+2)
        for j in edges:
          for k in range(len(PRIMITIVES[start+j])):
            gene.append((PRIMITIVES[start+j][k], j))
        start = end
        n += 1

      return gene

    gene_normal = _parse(True)
    gene_reduce = _parse(False)

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype