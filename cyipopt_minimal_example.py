# run this with the torch env john

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from functorch import vmap, vjp

_ = torch.manual_seed(0)

# double integrator dynamics
A = torch.tensor([
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])

B = torch.tensor([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0]
])

linear_f = lambda x, u: x @ A.T + u @ B.T
quadratic_f = lambda x, u: linear_f(x,u) + x.T @ x
x = torch.randn([6])
u = torch.randn([3])

Jac = torch.func.jacrev(quadratic_f, (0,1))

Hess = torch.func.hessian(quadratic_f, (0,))