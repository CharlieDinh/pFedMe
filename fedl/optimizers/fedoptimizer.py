from torch.optim import Optimizer


class MySGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
        return loss


class FEDLOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, server_grads=None, pre_grads=None, eta=0.1):
        self.server_grads = server_grads
        self.pre_grads = pre_grads
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, eta=eta)
        super(FEDLOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            i = 0
            for p in group['params']:
                p.data = p.data - group['lr'] * \
                         (p.grad.data + group['eta'] * self.server_grads[i] - self.pre_grads[i])
                # p.data.add_(-group['lr'], p.grad.data)
                i += 1
        return loss

class PersionalizedOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda)
        super(PersionalizedOptimizer, self).__init__(params, defaults)
    
    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data))
        return  group['params'], loss
    
    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = localweight.data
        #return  p.data
        return  group['params']


class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, closure=None, alpha = 1, n_k = 1):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = alpha  * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)
        return loss
