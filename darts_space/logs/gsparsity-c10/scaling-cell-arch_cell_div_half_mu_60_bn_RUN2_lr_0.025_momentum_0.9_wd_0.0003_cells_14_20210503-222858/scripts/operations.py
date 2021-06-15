import torch
import torch.nn as nn

OPS = {
  'avg_pool_3x3' : lambda C, stride, affine: nn.Sequential(AvgPoolBN(C, stride, affine=affine), ScaleLayer(affine)),
  'max_pool_3x3' : lambda C, stride, affine: nn.Sequential(MaxPoolBN(C, stride, affine=affine), ScaleLayer(affine)),
  'skip_connect' : lambda C, stride, affine: nn.Sequential(Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine), ScaleLayer(affine)),
  'sep_conv_3x3' : lambda C, stride, affine: nn.Sequential(SepConv(C, C, 3, stride, 1, affine=affine)),
  'sep_conv_5x5' : lambda C, stride, affine: nn.Sequential(SepConv(C, C, 5, stride, 2, affine=affine)),
  'sep_conv_7x7' : lambda C, stride, affine: nn.Sequential(SepConv(C, C, 7, stride, 3, affine=affine)),
  'dil_conv_3x3' : lambda C, stride, affine: nn.Sequential(DilConv(C, C, 3, stride, 2, 2, affine=affine)),
  'dil_conv_5x5' : lambda C, stride, affine: nn.Sequential(DilConv(C, C, 5, stride, 4, 2, affine=affine)),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    )),
}

class ScaleLayer(nn.Module):
    def __init__(self, affine, init_value=1):
        super().__init__()
        if affine: # "affine" is set to be True in evaluation mode
            self.scale = 1
#             print("evaluation mode in ScaleLayer: {}".format(self.scale))
        else: # "affine" is set to be False in search mode
            self.scale = nn.Parameter(torch.FloatTensor([init_value]))
#             print("search mode in ScaleLayer: {}".format(self.scale))

    def forward(self, input):
        return input * self.scale

class ScaleLayer_vec(nn.Module):
    def __init__(self, affine, init_value=1):
        super().__init__()
        if affine: # "affine" is set to be True in evaluation mode
            self.scale = 1
#             print("evaluation mode in ScaleLayer: {}".format(self.scale))
        else: # "affine" is set to be False in search mode
            vec_dim = 6400
            self.scale = nn.Parameter(init_value / (vec_dim ** 0.5) * torch.ones(vec_dim))
#             print("search mode in ScaleLayer: {}".format(self.scale))

    def forward(self, input):
        return input * torch.norm(self.scale)
    
class AvgPoolBN(nn.Module):
  def __init__(self, C_out, stride, affine=True):
    super(AvgPoolBN, self).__init__()
    self.op = nn.Sequential(nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
                            nn.BatchNorm2d(C_out, affine=affine)
                           )

  def forward(self, x):
    return self.op(x)


class MaxPoolBN(nn.Module):
  def __init__(self, C_out, stride, affine=True):
    super(MaxPoolBN, self).__init__()
    self.op = nn.Sequential(nn.MaxPool2d(3, stride=stride, padding=1),
                            nn.BatchNorm2d(C_out, affine=affine)
                           )

  def forward(self, x):
    return self.op(x)

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class IdentityBN(nn.Module):
  def __init__(self, C, affine=True):
    super(IdentityBN, self).__init__()
    print("affine in IdentityBN: {}".format(affine))
    self.op = nn.Sequential(nn.BatchNorm2d(C, affine=affine))

  def forward(self, x):
    return self.op(x)

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

