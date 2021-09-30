'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchsummary import summary
from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,  in_planes, planes, blk, stride=1, option='A'):
        super(BasicBlock, self).__init__()

        
        self.conv1 = nn.Conv2d(in_planes, blk['conv1'], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(blk['conv1'])
        self.conv2 = nn.Conv2d(blk['conv1'], planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, out_filters, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], out_filters['layer1'], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], out_filters['layer2'], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], out_filters['layer3'], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, out_filters, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for blk, stride in zip(out_filters.items(), strides):
            layers.append(block(self.in_planes, planes, blk[1], stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
       
        out = F.avg_pool2d(out, int(out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56(out_filters, num_blocks):
    model = ResNet(BasicBlock, num_blocks, out_filters)
    return model


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":

    out_filter = {
        "layer1": {
            "0": {
                "in":7,
                "conv1": 13,
                "conv2": 14
            },
            "1": {
                "in":14,
                "conv1": 8,
                "conv2": 12
            },
            "2": {
                "in":12,
                "conv1": 13,
                "conv2": 9
            },
            "3": {
                "in":9,
                "conv1": 13,
                "conv2": 9
            },
            "4": {
                "in":9,
                "conv1": 11,
                "conv2": 6
            },
            "5": {
                "in":6,
                "conv1": 14,
                "conv2": 5
            },
        },
        "layer2": {
            "0": {
                "in":5,
                "conv1": 23,
                "conv2": 31
            },
            "1": {
                "in":31,
                "conv1": 32,
                "conv2": 27
            },
            "2": {
                "in":27,
                "conv1": 32,
                "conv2": 26
            },
            "3": {
                "in":26,
                "conv1": 29,
                "conv2": 24
            },            
        },
        "layer3": {
            "5": {
                "in":24,
                "conv1": 1,
                "conv2": 1
            },
            "6": {
                "in":1,
                "conv1": 57,
                "conv2": 34
            },
            "7": {
                "in":34,
                "conv1": 59,
                "conv2": 10
            },
            "8": {
                "in":10,
                "conv1": 7,
                "conv2": 7
            }
        }
    }
    
    model = resnet56(out_filter)
    model.cuda()
    summary(model, (3, 32, 32))
