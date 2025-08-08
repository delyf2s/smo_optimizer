import torch
from torch.optim import Optimizer

class SMO(Optimizer):
    """
    Selective Momentum Optimizer.

    Dynamically adjusts learning rate based on batch composition,
    combining momentum and decoupled weight decay.
    """

    def __init__(self, params, lr_true=1e-3, lr_false=1e-4, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr_true=lr_true,
                        lr_false=lr_false,
                        momentum=momentum,
                        weight_decay=weight_decay)
        super(SMO, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, y):
        """
        Performs a single optimization step.

        Args:
            y (Tensor): Tensor of labels or indicators (0 or 1) with shape [batch_size].
        """
        true_ratio = y.sum().item() / len(y)

        for group in self.param_groups:
            lr = group['lr_true'] * true_ratio + group['lr_false'] * (1 - true_ratio)
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)

                p.add_(buf, alpha=-lr)
