import torch
from torch.optim.optimizer import Optimizer, required


class ProxSGD(Optimizer):
    """Implementation of Prox-SGD algorithm, proposed in
        "Prox-SGD: Training Structured Neural Networks under Regularization and Constraints", ICLR 2020.
        https://openreview.net/forum?id=HygpthEtvr

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): initial learning rate (default: 1e-3)
        momentum (float, optional): initial momentum value used for computing running averages of gradient (default:  0.9)
        mu (float, optional): regularization constant mu (for L1 penalty) (default: 1e-4)
    """

    def __init__(self, params, lr=0.06, momentum=0.9, mu=None, clip_bounds=(None,None)):
        if not 0.0 <= lr:
            raise ValueError("Invalid initial learning rate: {}".format(lr))
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("Invalid momentum parameter: {}".format(momentum))

        defaults = dict(lr=lr, momentum=momentum, mu=mu, clip_bounds=clip_bounds)
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

                v_t = state['v_t']

                state['step'] += 1
                

                lr_t = group['lr']
                momentum_t = group['momentum']

                # Decay the first and second moment running average coefficient
                v_t.mul_(1 - momentum_t).add_(momentum_t, grad)

                x_tmp = x - v_t

                if group['mu'] is not None:
                    mu_t  = group['mu']
                    x_hat = torch.max(x_tmp - mu_t, torch.zeros_like(x_tmp)) - torch.max(-x_tmp - mu_t, torch.zeros_like(x_tmp))
                else:
                    x_hat = x_tmp

                low, up = group['clip_bounds']
                if low is not None or up is not None:
                    x_hat = x_hat.clamp_(low,up)

                x.data.add_(lr_t, x_hat - x)
        return loss