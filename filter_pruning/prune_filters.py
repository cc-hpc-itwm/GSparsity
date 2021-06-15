#This file is used to prune the model based on quality parameter which decides the threshold value by multiplying it with standard deviation of each layer.

import torch
import torch.nn as nn
import numpy as np

def prune_model(model,quantile_per):
    for index,module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            weight_copy=module.weight.data.clone()
            channel_norms=torch.norm((weight_copy.view(weight_copy.shape[0],-1)),p=2,dim=1)
            channel_norms=channel_norms.gt(quantile_per).float().cuda()
            channel_norms=channel_norms.reshape(channel_norms.shape[0],1)
            module.weight.data=torch.einsum('ijkl, im -> ijkl',weight_copy,channel_norms)
    return model