# Code taken and modified from Stochastic Frank-Wolfe for Constrained Optimization

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
#from frankwolfe.feasible_regions import nuclear_norm_ball
from scipy.sparse.linalg import svds
import numpy as np


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
    (n,) = vect.shape  # will raise ValueError if v is not 1-D
    if vect.sum() == s and np.alltrue(vect >= 0):
        return vect
    v = vect - np.max(vect)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.count_nonzero(u * np.arange(1, n + 1) > (cssv - s)) - 1
    theta = float(cssv[rho] - s) / (rho + 1)
    w = (v - theta).clip(min=0)
    return w

@torch.no_grad()
def make_feasible(model, constraints):
    """Shift all model parameters inside the feasible region defined by constraints"""
    for idx, (name, param) in enumerate(model.named_parameters()):
        constraint = constraints[idx]
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

class NuclearNormBall(Constraint):
    def __init__(self, dim1, dim2, alpha=1.0):
        super().__init__(n=dim1 * dim2)
        self.dim1 = dim1
        self.dim2 = dim2
        self.alpha = alpha
        
    def lmo(self, x):
        super().lmo(x)
        x = x.view(self.dim1, self.dim2)
        x = x.detach().cpu().numpy()
        u, s, vt = svds(-x, k=1, which='LM')
        return torch.tensor(self.alpha * u @ vt)
    
    def euclidean_project(self, x):
        super().euclidean_project(x)
        x = x.view(self.dim1, self.dim2)
        u, s, vt = torch.linalg.svd(x, full_matrices=False)
        return u,s,vt
        

x = torch.randn(6000, 3000)

nuclear_norm = NuclearNormBall(6000, 3000, 100)
v = nuclear_norm.lmo(x)
v_proj = nuclear_norm.euclidean_project(x)

class LpBall(Constraint):
    """
    LMO class for the n-dim Lp-Ball (p=ord) with L2-diameter diameter or radius.
    """

    def __init__(self, n, ord=2, diameter=None, radius=None):
        super().__init__(n)
        self.p = float(ord)
        self.q = get_lp_complementary_order(self.p)

        assert float(ord) >= 1, f"Invalid order {ord}"
        if diameter is None and radius is None:
            raise ValueError("Neither diameter nor radius given.")
        elif diameter is None:
            self._radius = radius
            self._diameter = 2 * convert_lp_radius(radius, self.n, in_ord=self.p, out_ord=2)
        elif radius is None:
            self._radius = convert_lp_radius(diameter / 2.0, self.n, in_ord=2, out_ord=self.p)
            self._diameter = diameter
        else:
            raise ValueError("Both diameter and radius given")

    @torch.no_grad()
    def lmo(self, x):
        """Returns v with norm(v, self.p) <= r minimizing v*x"""
        super().lmo(x)
        if self.p == 1:
            v = torch.zeros_like(x)
            maxIdx = torch.argmax(torch.abs(x))
            v.view(-1)[maxIdx] = -self._radius * torch.sign(x.view(-1)[maxIdx])
            return v
        elif self.p == 2:
            x_norm = float(torch.norm(x, p=2))
            if x_norm > tolerance:
                return -self._radius * x.div(x_norm)
            else:
                return torch.zeros_like(x)
        elif self.p == float('inf'):
            return torch.full_like(x, fill_value=self._radius).masked_fill_(x > 0, -self._radius)
        else:
            sgn_x = torch.sign(x).masked_fill_(x == 0, 1.0)
            absxqp = torch.pow(torch.abs(x), self.q / self.p)
            x_norm = float(torch.pow(torch.norm(x, p=self.q), self.q / self.p))
            if x_norm > tolerance:
                return -self._radius / x_norm * sgn_x * absxqp
            else:
                return torch.zeros_like(x)

    @torch.no_grad()
    def shift_inside(self, x):
        """Projects x to the LpBall with radius r.
        NOTE: This is a valid projection, although not the one mapping to minimum distance points.
        """
        super().shift_inside(x)
        x_norm = torch.norm(x, p=self.p)
        return self._radius * x.div(x_norm) if x_norm > self._radius else x

    @torch.no_grad()
    def euclidean_project(self, x):
        """Projects x to the closest (i.e. in L2-norm) point on the LpBall (p = 1, 2, inf) with radius r."""
        super().euclidean_project(x)
        if self.p == 1:
            x_norm = torch.norm(x, p=1)
            if x_norm > self._radius:
                sorted = torch.sort(torch.abs(x.flatten()), descending=True).values
                running_mean = (torch.cumsum(sorted, 0) - self._radius) / torch.arange(1, sorted.numel() + 1,
                                                                                       device=x.device)
                is_less_or_equal = sorted <= running_mean
                # This works b/c if one element is True, so are all later elements
                idx = is_less_or_equal.numel() - is_less_or_equal.sum() - 1
                return torch.sign(x) * torch.max(torch.abs(x) - running_mean[idx], torch.zeros_like(x))
            else:
                return x
        elif self.p == 2:
            x_norm = torch.norm(x, p=2)
            return self._radius * x.div(x_norm) if x_norm > self._radius else x
        elif self.p == float('inf'):
            return torch.clamp(x, min=-self._radius, max=self._radius)
        else:
            raise NotImplementedError(f"Projection not implemented for order {self.p}")

    

class StochasticFrankWolfe(Optimizer):
    def __init__(self, *args, **kwargs):
        pass
    
    def step(self, constraint, closure=None):
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
                if momentum > 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = d_p.detach().clone()
                    else:
                        param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - momentum)
                v = constraint.linear_optimization_oracle(p.grad)
                if self.rescale = 'diameter':
                    factor = 1./constraint.
