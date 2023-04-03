import torch
from torch.optim.optimizer import Optimizer, required


def soft_threshold(x, gamma):
    y = torch.max(x.new_tensor(0), torch.abs(x) - gamma)
    return torch.sign(x) * y

class APG(Optimizer):

    def __init__(self, params, lr=required, momentum=0, weight_decay=0, gamma=0.9):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if gamma < 0.0:
            raise ValueError("Invalid gamma value: {}".format(gamma))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, gamma=gamma)
        super(APG, self).__init__(params, defaults)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            gamma = group['gamma']

            for p in group['params']:
                if p.grad is None:
                    continue
                ratio = p.ratio
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                z = p.data.add(-group['lr'], d_p)
                z = soft_threshold(z, group['lr'] * gamma * ratio)
                new_v = z - p.data

                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(new_v).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(new_v)

                p.data = z + momentum * buf
                #no negtive
                p.data = torch.max(p.data, torch.zeros(len(p.data), device=p.data.device))

        return loss
