import torch

class LBFGSWithCounter(torch.optim.LBFGS):
    def __init__(self, *args, **kwargs):
        super(LBFGSWithCounter, self).__init__(*args, **kwargs)
        self.num_iterations = 0
    def step(self, closure):
        result = super(LBFGSWithCounter, self).step(closure)
        return result