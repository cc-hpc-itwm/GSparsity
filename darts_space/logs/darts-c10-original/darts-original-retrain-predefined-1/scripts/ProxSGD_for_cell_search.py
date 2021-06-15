import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class ProxSGD(Optimizer):
    r"""Implements Prox-SGD algorithm.

    It has been proposed in `Prox-SGD: Training Structured Neural Networks under Regularization and Constraints`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        epsilon_decay (float, optional): decay factor used for decaying the learning rate over time
        rho_decay (float, optional): decay factor used for decaying the momentum term over time
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        delta (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): regularization constant (L1 penalty) (default: 1e-4)
        gamma (integer, optional): offset in time for decaying the learning rate as well as the momentum term
            (default: 4)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=0.06, epsilon_decay=0.5, rho_decay=0.5, betas=(0.9, 0.999), delta=1e-8,
                 weight_decay=None, gamma=4, clip_bounds=(None,None), normalization=False, normalization_exponent=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= delta:
            raise ValueError("Invalid delta value: {}".format(delta))
        if not 0.0 <= epsilon_decay < 1.0:
            raise ValueError("Invalid epsilon decay parameter at index 0: {}".format(epsilon_decay))
        if not 0.0 <= rho_decay < 1.0:
            raise ValueError("Invalid rho decay parameter at index 1: {}".format(rho_decay))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.normalization = normalization
        self.normalization_exponent = normalization_exponent
        defaults = dict(lr=lr, epsilon_decay=epsilon_decay, rho_decay=rho_decay, betas=betas, delta=delta,
                        weight_decay=weight_decay, gamma=gamma, clip_bounds=clip_bounds)
        super(ProxSGD, self).__init__(params, defaults)      

    def __setstate__(self, state):
        super(ProxSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

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
            b_group = 0
            if self.normalization:
                dim_group = (torch.zeros(1)).cuda()
            else: # assuming "multiply" normalization
                dim_group = (torch.ones(1)).cuda()
                
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
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['v_t'] = torch.zeros_like(x.data)

                v_t = state['v_t']

                state['step'] += 1

                rho, beta = group['betas']

                rho_t = rho

                # Decay the first and second moment running average coefficient
                v_t.mul_(1 - rho_t).add_(rho_t, grad)

                tau = 1
                b  = x - v_t / tau          

                if group['weight_decay'] is not None:
                    b_group += torch.norm(b)**2                
                    if self.normalization:
                        dim_group += torch.numel(x)

            for x in group['params']:
                #print(len(x.size()),x.size()[0])
                if x.grad is None:
                    continue
                grad = x.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Prox-SGD does not support sparse gradients')

                state = self.state[x]

                v_t = state['v_t'] # the momentum v_t is updated in the loop above

                epsilon = group['lr']
                
                tau = 1
                b  = x - v_t / tau          

                if group['weight_decay'] is not None:
                    a  = torch.pow(dim_group, self.normalization_exponent) * group['weight_decay'] / tau
                    x_hat = torch.clamp(1-a/torch.sqrt(b_group), min=0)*b
                else:
                    x_hat = b

#                 low, up = group['clip_bounds']
#                 if len(x.size()) == 1 and x.size()[0] == 1:
#                     if low is not None or up is not None:
#                         x_hat = x_hat.clamp_(low,up)

                x.data.add_(epsilon,x_hat-x)
                
        return loss