import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import Genotype

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
"""
class MixedOp(nn.Module):

  def __init__(self, C, stride, primitives):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in primitives:
      op = OPS[primitive](C, stride, False)
      self._ops.append(op)

  def forward(self, x):
    return sum(op(x) for op in self._ops)
"""

class MixedOp(nn.Module):

  def __init__(self, C, stride, primitives, k):
    super(MixedOp, self).__init__()
    self.k = k
    self.C = C
    self._ops = nn.ModuleList()
    self.mp = nn.MaxPool2d(2,2)
    for primitive in primitives:
      op = OPS[primitive](C //self.k, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C //self.k, affine=False))
      self._ops.append(op)

  def forward(self, x):
    #channel proportion k=4  
    dim_2 = x.shape[1]
    xtemp = x[ : , :  dim_2//self.k, :, :]
    xtemp2 = x[ : ,  dim_2//self.k:, :, :]
    temp1 = sum(op(xtemp) for op in self._ops)
    #reduction cell needs pooling before concat
    if temp1.shape[2] == x.shape[2]:
      ans = torch.cat([temp1,xtemp2],dim=1)
    else:
      ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
    ans = channel_shuffle(ans,self.k)
    #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
    #except channe shuffle, channel shift also works
    return ans

  def wider(self, k):
    self.k = k
    for op in self._ops:
      op.wider(self.C//k, self.C//k)


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduce, reduce_prev, primitives_cell,k):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C, reduce, reduce_prev)
    self.reduce = reduce
    self.k = k

    if reduce_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps # number of intermediate nodes
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()

    edge_index = 0

    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduce and j < 2 else 1 # the first two input nodes have a stride of 2
        op = MixedOp(C, stride, primitives_cell[edge_index],self.k)
        self._ops.append(op)
        edge_index += 1

  def forward(self, s0, s1):
    #print("Input to preprocess: ",s0.size(),s1.size())
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    #print(self.preprocess0)
    #print("Inside Cell: ",s0.size(),s1.size())

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h) for j, h in enumerate(states))
      #print(s.size())
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)

  def wider(self, k):
    self.k = k
    for op in self._ops:
      op.wider(k)

class Network(nn.Module):

  def __init__(self, C, num_classes, cells, criterion, primitives, steps=4,
               multiplier=4, stem_multiplier=3, k=4):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._cells = cells
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    
    self.primitives = primitives

    C_curr = stem_multiplier*C
    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr // 2, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduce_prev = True#False
    for i in range(cells):
      if i in [cells//3, 2*cells//3]:
        C_curr *= 2
        reduce = True
        primitives_cell = primitives['primitives_reduct']
      else:
        reduce = False
        primitives_cell = primitives['primitives_normal']
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduce, reduce_prev, primitives_cell, k)
      reduce_prev = reduce
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    #s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      #print(i)
      #print(s0.size(),s1.size())
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
        #print(edges)
        for j in edges:
          k_best = None
          for k in range(len(PRIMITIVES[j])):
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