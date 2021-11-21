import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

def plot_individual_op_norm(model, filename, normalization="none", normalization_exponent=0):
    """plot the norm of each operation in the given model"""
    
    op_norms = [torch.norm(p) for p in model.parameters() if p.requires_grad]
    op_names = [q for q, p in model.named_parameters() if p.requires_grad]
    op_sizes = [p.numel() for p in model.parameters() if p.requires_grad]
        
    num_ops = len(op_names)
    f = plt.figure(num=None, figsize=(num_ops*0.15, 6), dpi=100, facecolor='w', edgecolor='k')
    if normalization == "none":
        for i, op_norm in enumerate(op_norms):
            plt.semilogy(i, op_norm.item(),"o")
    elif normalization == "mul":
        for i, (op_norm, op_size) in enumerate(zip(op_norms, op_sizes)):
            op_norm_normalized = op_norm * (op_size ** normalization_exponent)
            plt.semilogy(i, op_norm_normalized.item(),"o")
    elif normalization == "div":
        for i, (op_norm, op_size) in enumerate(zip(op_norms, op_sizes)):
            op_norm_normalized = op_norm / (op_size ** normalization_exponent)
            plt.semilogy(i, op_norm_normalized.item(),"o")

    plt.xticks(np.arange(num_ops), op_names, rotation=90)
    plt.xlim(-1, num_ops)    
    plt.ylim(1e-5, 1e5)    
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
