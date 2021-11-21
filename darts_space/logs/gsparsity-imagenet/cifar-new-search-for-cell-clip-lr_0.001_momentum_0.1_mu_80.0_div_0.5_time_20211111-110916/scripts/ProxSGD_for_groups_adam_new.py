import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np
import math

class ProxSGD(Optimizer):
    r"""Implements Prox-SGD algorithm for pruning 'groups'. 
    
    A group could be the same operation in all cells of the same type (normal or reduce). For example, one group is op 3 of edge 7 in Cells 0-1/3-4/5-7 (normal cells), another group is op 3 of edge 7 in Cell 2/5 (reduce cells).
    
    A group could also be the same operation in all cells in a stage (there may be multiple stages). For example, one group is op 3 of edge 7 in all cells of stage_normal_1 (Cells 0-1), another group is op 3 of edge 7 in all cells of stage_reduce_1 (Cell 2), another group is op 3 of edge 7 in all cells of stage_normal_2 (Cells 3-4), another group is op 3 of edge 7 in all cells of stage_reduce_2 (Cell 5), one group is op 3 of edge 7 in all cells of stage_normal_3 (Cells 6-7).
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999),
                 weight_decay=None, clip_bounds=(None,None), normalization="none", normalization_exponent=0, eps=1e-6):
        if not 0.0 <= lr <= 1.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        #if not 0.0 <= momentum <= 1.0:
        #    raise ValueError("Invalid momentum parameter: {}".format(momentum))
        #if weight_decay is not None and weight_decay < 0:
        #    raise ValueError("Invalid weight decay parameter: {}".format(momentum))
        if not 0.0 <= normalization_exponent:
            raise ValueError("Invalid normalization exponent parameter: {}".format(momentum))

        self.normalization = normalization
        self.normalization_exponent = normalization_exponent
        defaults = dict(lr=lr, betas=betas, clip_bounds=clip_bounds, eps=eps)
        super(ProxSGD, self).__init__(params, defaults)      

    def __setstate__(self, state):
        super(ProxSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            #group['lr'] = lr

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            b_group_norm = torch.zeros(1).cuda()
            tau_group = torch.zeros(1).cuda()
            if self.normalization == "none":
                dim_group = (torch.ones(1)).cuda()
            else: # "mul" or "div" normalization
                dim_group = (torch.zeros(1)).cuda()
                
            for x in group['params']:
                #print(len(x.size()),x.size()[0])
                if x.grad is None:
                    continue
                grad = x.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Prox-SGD does not support sparse gradients')

                state = self.state[x]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(x.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(x.data)
                    # Exponential moving average of squared gradient values
                    state["tau"] = torch.zeros_like(torch.mean(x.data))

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg.mul_(1-beta1).add_(grad, alpha=beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                #exp_avg_sq_group += exp_avg_sq
                #denom = exp_avg_sq.sqrt().add_(group["eps"])

                #momentum = group['momentum']
                # Decay the first and second moment running average coefficient
                #v_t.mul_(1 - momentum).add_(momentum, grad)

                #step_size = group["lr"]
                
                #bias_correction1 = 1.0 - beta1 ** state["step"]
                #print(exp_avg_sq)
                
                bias_correction2 = 1.0 - beta2 ** state["step"]

                #if group['weight_decay'] is not None:
                    
                tau_group += torch.sum(torch.sqrt(exp_avg_sq/bias_correction2).add_(group["eps"]))#torch.norm(torch.sqrt(exp_avg_sq/bias_correction2).add_(group["eps"]))**2
                dim_group += torch.numel(x)
                #state["tau"] = tau
            #print(tau_group)
            #if group['weight_decay'] is not None:
            tau_group = tau_group/dim_group#torch.sqrt(tau_group)
            #else:
            #tau_group = torch.ones(1).cuda()
            #print(tau_group)

            for x in group['params']:


                if x.grad is None:
                    continue
                grad = x.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Prox-SGD does not support sparse gradients')

                state = self.state[x]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                #tau = state["tau"]
                beta1, beta2 = group["betas"]

                b  = x - exp_avg / tau_group    

                state["b_val"] = b      

                if group['weight_decay'] is not None:
                    b_group_norm += torch.norm(b)**2
                    #dim_group += torch.numel(x)

            b_group_norm = torch.sqrt(b_group_norm)
            
            
            for x, x_is_scale_op in zip(group['params'], group['scale']):
                #print(len(x.size()),x.size()[0])
                if x.grad is None:
                    continue
                grad = x.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Prox-SGD does not support sparse gradients')

                state = self.state[x]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                tau = state["tau"]
                beta1, beta2 = group["betas"]


                step_size = group["lr"]
                
                bias_correction1 = 1.0 - beta1 ** state["step"]
                #bias_correction2 = 1.0 - beta2 ** state["step"]
                
                step_size = step_size / bias_correction1

                
                #tau = tau_group #1 / np.sqrt()
                b  = state["b_val"]#x - exp_avg / tau          


                if group['weight_decay'] is not None: # operations are prunable
                    if self.normalization == "none": # no normalization
                        a  = group['weight_decay'] / tau_group
                    elif self.normalization == "mul": # normalize the weight decay by multiplying operation_dimension**exponent
                        a  = group['weight_decay'] * torch.pow(dim_group, self.normalization_exponent) / tau_group
                    elif self.normalization == "div": # normalize the weight decay by dividing operation_dimension**exponent
                        a  = group['weight_decay'] / torch.pow(dim_group, self.normalization_exponent) / tau_group #* reg_fac
                        
                    x_hat = torch.clamp(1 - a / b_group_norm, min=0) * b
                    
                    low, up = group['clip_bounds']
                    if x_is_scale_op:
                        if low is not None or up is not None:
                            x_hat = x_hat.clamp_(low, up)
                else: #operations are nonprunable
                    x_hat = b


                x.data.add_(step_size, x_hat - x)
                
        return loss