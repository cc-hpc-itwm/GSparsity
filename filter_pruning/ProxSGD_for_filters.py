import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import gc


class ProxSGD(Optimizer):
    """ProxSGD algorithm for filter purning.
    
    It is an instantiation of the ProxSGD algorithm proposed in
    `ProxSGD: Training Structured Neural Networks under Regularization and Constraints`
    (https://openreview.net/forum?id=HygpthEtvr)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
                           parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): momentum (default: 0.9)
        
        delta (float, optional): term added to the denominator to improve
                                 numerical stability (default: 1e-8)
        beta (float, optional): beta coefficient used for computing
                                running averages of the square of the gradient (default:  0.999)
        For the rationale of delta and beta, refer to:
        Adam: A Method for Stochastic Optimization (https://arxiv.org/abs/1412.6980).
        
    """

    def __init__(self, params, lr=0.001, momentum=0.9, delta=1e-8, beta=0.999, adaptive_lr=False, normalization=None):
        if not 0.0 <= lr <= 1:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum <= 1:
            raise ValueError("Invalid momentum: {}".format(momentum))
        if not 0.0 <= delta:
            raise ValueError("Invalid delta value: {}".format(delta))
        if not 0.0 <= beta < 1:
            raise ValueError("Invalid beta value: {}".format(beta))

        defaults = dict(lr=lr,
                        momentum=momentum,
                        delta=delta,
                        beta=beta,
                        adaptive_lr=adaptive_lr)
        
        self.normalization = normalization
        
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

                lr = group['lr']
                momentum = group['momentum']
                beta = group['beta']

                state = self.state[x]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['v_t'] = torch.zeros_like(x.data)
                    # Exponential moving average of squared gradient values
                    state['r_t'] = torch.zeros_like(x.data)

                # Decay the first and second moment running average coefficient
                v_t = state['v_t']
                v_t.mul_(1 - momentum).add_(momentum, grad)

                r_t = state['r_t']
                r_t.mul_(beta).addcmul_(1 - beta, grad, grad)

                x_shape, num_filters = x.shape, x.shape[0]
                
                if group['adaptive_lr']:
                    state['step'] += 1
                    bias_correction = 1 - beta ** (state['step'])

                    tau = (r_t/bias_correction).sqrt().add_(group['delta'])
                    tau = tau.view(num_filters, -1)
                    tau = torch.mean(tau, 1)

                    tau_shape = [num_filters]
                    for _ in range(1, len(x_shape)):
                        tau_shape.append(1)

                    tau_tensor = tau.reshape(tau_shape)
                else:
                    tau = 1
                    tau_tensor = 1

                if group['weight_decay'] is not None:  # proximal operation for prunable filters
                    x_tilde = x - v_t / tau_tensor

                    # reshape tensor from shape of [#filters * #in_filters * kernel_size]  to [#filters * (#in_filters * filter_size)]
                    x_tilde = x_tilde.view(num_filters, -1)
                    x_tilde_norm = torch.norm(x_tilde, p=2, dim=1)  # L2 Norm of filters (shape of x_tilde_norm: [#filters])

                    filter_size = torch.tensor(np.prod([v for v in x_shape])/num_filters).float()

                    if self.normalization is None:
                        normalized_mu = group['weight_decay'] / tau
                    elif self.normalization == "mul":
                        normalized_mu = group['weight_decay'] * torch.sqrt(filter_size) / tau
                    elif self.normalization == "div":
                        normalized_mu = group['weight_decay'] / torch.sqrt(filter_size) / tau
                    
                    thresholding = torch.clamp(1-normalized_mu/x_tilde_norm, min=0)
                    thresholding = thresholding.view(num_filters, -1)  # reshape tensor to 1D [#filters * 1]
                    
                    x_proximal = torch.mul(thresholding, x_tilde)
                    x_proximal = x_proximal.view(x_shape)  # reshape back to original weight tensor shape

                    x.data.add_(lr, x_proximal-x)
                else:  # other parameters that are not prunable
                    x.data.add_(lr, -v_t/tau_tensor)

        gc.collect()
        return loss
