import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class ProxSGD(Optimizer):
    r"""Prox-SGD algorithm for purning operations.

    It has been proposed in `Prox-SGD: Training Structured Neural Networks under Regularization and Constraints`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        delta (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): regularization constant (L1 penalty) (default: 1e-4)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=0.06, momentum=0.9, weight_decay=None, delta=1e-8, beta=0.999, clip_bounds=(None, None)):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= delta:
            raise ValueError("Invalid delta value: {}".format(delta))

        defaults = dict(lr=lr, momentum=momentum, delta=delta,
                        weight_decay=weight_decay, beta=beta, clip_bounds=clip_bounds)
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
                    # Exponential moving average of squared gradient values
                    state['r_t'] = torch.zeros_like(x.data)

                v_t, r_t = state['v_t'], state['r_t']

                state['step'] += 1

                momentum = group['momentum']
                beta = group['beta']

                lr = group['lr']

                # Decay the first and second moment running average coefficient
                v_t.mul_(1 - momentum).add_(momentum, grad)
                r_t.mul_(beta).addcmul_(1 - beta, grad, grad)

                bias_correction = 1 - beta ** (state['step'])

                #tau = (torch.mean(r_t) / bias_correction).sqrt().add_(group['delta'])
                tau = (r_t / bias_correction).sqrt().add_(group['delta'])
                tau = torch.mean(tau)
                b  = x - v_t / tau          

                if group['weight_decay'] is not None:
                    a  = group['weight_decay'] / tau 
                    x_hat = torch.clamp(1 - a / torch.norm(b), min=0) * b
                else:
                    x_hat = b

#                 low, up = group['clip_bounds']
#                 if len(x.size()) == 1 and x.size()[0] == 1:
#                     if low is not None or up is not None:
#                         x_hat = x_hat.clamp_(low,up)

                x.data.add_(lr, x_hat - x)

        return loss