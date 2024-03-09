# Code taken and modified from Stochastic Frank-Wolfe for Constrained Optimization

import torch
import torch.nn.functional as F
from torch.optim import Optimizer, SGD
from scipy.sparse.linalg import svds
import numpy as np

device = torch.device("mps")

@torch.no_grad()
def get_avg_init_norm(layer, param_type=None, ord=2, repetitions=100):
    """Computes the average norm of default layer initialization"""
    output = 0
    for _ in range(repetitions):
        layer.reset_parameters()
        output += torch.norm(getattr(layer, param_type), p=ord).item()
    return float(output) / repetitions


def get_lp_complementary_order(ord):
    """Get the complementary order"""
    ord = float(ord)
    if ord == float('inf'):
        return 1
    elif ord == 1:
        return float('inf')
    elif ord >= 2:
        return 1.0 / (1.0 - 1.0 / ord)
    else:
        raise NotImplementedError(f"Order {ord} not supported.")
    
    
def convert_lp_radius(r, N, in_ord=2, out_ord='inf'):
    """
    Convert between radius of Lp balls such that the ball of order out_order
    has the same L2 diameter as the ball with radius r of order in_order
    in N dimensions
    """
    # Convert 'inf' to float('inf') if necessary
    in_ord, out_ord = float(in_ord), float(out_ord)
    in_ord_rec = 0.5 if in_ord == 1 else 1.0 / in_ord
    out_ord_rec = 0.5 if out_ord == 1 else 1.0 / out_ord

    return r * N ** (out_ord_rec - in_ord_rec)



def projection_simplex_sort(vect, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n = vect.shape[0]  # will raise ValueError if v is not 1-D
    if vect.sum() == s and torch.all(vect >= 0):
        return vect
    v = vect - torch.max(vect)
    u = torch.sort(v, descending=True)[0]
    cssv = torch.cumsum(u, dim=0)
    rho = torch.nonzero(u * torch.arange(1, n + 1).to(vect.device) > (cssv - s), as_tuple=True)[0].size(0) - 1
    theta = float(cssv[rho] - s) / (rho + 1)
    w = torch.clamp(v - theta, min=0)
    return w
    

@torch.no_grad()
def make_feasible(model, constraint):
    """Shift all model parameters inside the feasible region defined by constraints"""
    for idx, (name, param) in enumerate(model.named_parameters()):
        param.copy_(constraint.shift_inside(param))

class Constraint:
    """
    Parent/Base class for constraints
    :param n: dimension of constraint parameter space
    """

    def __init__(self, n):
        self.n = n
        self._diameter, self._radius = None, None

    def is_unconstrained(self):
        return False

    def get_diameter(self):
        return self._diameter

    def get_radius(self):
        try:
            return self._radius
        except:
            raise ValueError("Tried to get radius from a constraint without one")

    def lmo(self, x):
        assert x.numel() == self.n, f"shape {x.shape} does not match dimension {self.n}"

    def shift_inside(self, x):
        assert x.numel() == self.n, f"shape {x.shape} does not match dimension {self.n}"

    def euclidean_project(self, x):
        assert x.numel() == self.n, f"shape {x.shape} does not match dimension {self.n}"
        
def nuclear_norm(matrix):
    norm = torch.norm(matrix, p='nuc')
    return norm

class NuclearNormBall(Constraint):
    def __init__(self, dim1, dim2, alpha=1.0):
        super().__init__(n=dim1 * dim2)
        self.dim1 = dim1
        self.dim2 = dim2
        self.alpha = alpha
        
    @torch.no_grad()
    def lmo(self, x):
        super().lmo(x)
        x = x.view(self.dim1, self.dim2)
        #print(x.shape)
        u, s , vt = torch.svd_lowrank(x, q=1)
        #print(u.shape, s.shape, vt.shape)
        #u_1 , s_1, vt_1 = u[:,0].unsqueeze(1), s[0], vt[0,:].unsqueeze(0)
        return -self.alpha * u @ vt.T
    
    @torch.no_grad()
    def shift_inside(self, x):
        return self.euclidean_project(x)
       
    # @torch.no_grad() 
    # def lmo(self, x):
    #     super().lmo(x)
    #     x = x.view(self.dim1, self.dim2)
    #     x = x.detach().cpu().numpy()
    #     u, s, vt = svds(-x, k=1, which='LM')
    #     return torch.from_numpy(self.alpha * u @ vt)
    
    @torch.no_grad()
    def euclidean_project(self, x):
        super().euclidean_project(x)
        x = x.view(self.dim1, self.dim2)
        u, s, vt = torch.linalg.svd(x, full_matrices=False)
        #print(u.shape, s.shape, vt.shape)
        s_ = projection_simplex_sort(s)
        return self.alpha * u @ torch.diag(s_) @ vt  

class VanillaFW(Optimizer):
    def __init__(self, params, learning_rate):
        if not (0.0 <= learning_rate <= 1.0):
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        
        default = dict(lr=learning_rate)
        super(VanillaFW, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, constraints, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                v = constraints[idx].lmo(d_p)
                lr = max(0.0, min(group['lr'], 1.0))
                p.mul_(1 - lr)
                p.add_(v, alpha=lr)
                idx += 1
        return loss
                    
        
class SFW(torch.optim.Optimizer):
    """Stochastic Frank Wolfe Algorithm
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        learning_rate (float): learning rate between 0.0 and 1.0
        rescale (string or None): Type of learning_rate rescaling. Must be 'diameter', 'gradient' or None
        momentum (float): momentum factor, 0 for no momentum
    """

    def __init__(self, params, learning_rate=0.1, rescale='diameter', momentum=0.9):
        if not (0.0 <= learning_rate <= 1.0):
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("Momentum must be between [0, 1].")
        if not (rescale in ['diameter', 'gradient', None]):
            raise ValueError("Rescale type must be either 'diameter', 'gradient' or None.")

        # Parameters
        self.rescale = rescale

        defaults = dict(lr=learning_rate, momentum=momentum)
        super(SFW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, constraints, closure=None):
        """Performs a single optimization step.
        Args:
            constraints (iterable): list of constraints, where each is an initialization of Constraint subclasses
            parameter groups
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                # Add momentum
                momentum = group['momentum']
                if momentum > 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = d_p.detach().clone()
                    else:
                        param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - momentum)
                        d_p = param_state['momentum_buffer']

                v = constraints[idx].lmo(d_p)  # LMO optimal solution

                if self.rescale == 'diameter':
                    # Rescale lr by diameter
                    factor = 1. / constraints[idx].get_diameter()
                elif self.rescale == 'gradient':
                    # Rescale lr by gradient
                    factor = torch.norm(d_p, p=2) / torch.norm(p - v, p=2)
                else:
                    # No rescaling
                    factor = 1

                lr = max(0.0, min(factor * group['lr'], 1.0))  # Clamp between [0, 1]

                p.mul_(1 - lr)
                p.add_(v, alpha=lr)
                idx += 1
        return loss



# class StochasticFrankWolfe(Optimizer):
#     def __init__(self, *args, **kwargs):
#         pass
    
#     def step(self, constraint, closure=None):
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()
#         idx = 0
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 d_p = p.grad
#                 if momentum > 0:
#                     param_state = self.state[p]
#                     if 'momentum_buffer' not in param_state:
#                         param_state['momentum_buffer'] = d_p.detach().clone()
#                     else:
#                         param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - momentum)
#                 v = constraint.linear_optimization_oracle(p.grad)
#                 if self.rescale = 'diameter':
#                     factor = 1./constraint.
