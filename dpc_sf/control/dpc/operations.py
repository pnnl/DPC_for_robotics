import torch
from neuromancer.dynamics import ode
from neuromancer.modules import blocks

# Functions:
def posVel2cyl(state, cyl, radius):
    x = state[:, 0:1]
    y = state[:, 2:3]
    xc = cyl[:, 0:1]
    yc = cyl[:, 1:2]

    dx = x - xc
    dy = y - yc

    # Calculate the Euclidean distance from each point to the center of the cylinder
    distance_to_center = (dx**2 + dy**2) ** 0.5
    
    # Subtract the radius to get the distance to the cylinder surface
    distance_to_cylinder = distance_to_center - radius

    xdot = state[:, 1:2]
    ydot = state[:, 3:4]

    # Normalize the direction vector (from the point to the center of the cylinder)
    dx_normalized = dx / (distance_to_center + 1e-10)  # Adding a small number to prevent division by zero
    dy_normalized = dy / (distance_to_center + 1e-10)

    # Compute the dot product of the normalized direction vector with the velocity vector
    velocity_to_cylinder = dx_normalized * xdot + dy_normalized * ydot

    return distance_to_cylinder, velocity_to_cylinder

class Dynamics(ode.ODESystem):
    def __init__(self, insize, outsize, x_std=0.0) -> None:
        super().__init__(insize=insize, outsize=outsize)

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

        self.f = lambda x, u: x @ A.T + u @ B.T
        self.in_features = insize
        self.out_features = outsize
        
        # noise definitions
        self.x_std = x_std

    def ode_equations(self, x, u):
        # x = xu[:,0:6]
        # u = xu[:,6:9]
        # add noise if required
        x = x + torch.randn(x.shape) * self.x_std
        return self.f(x,u)

class processP2PPolicyInput(torch.nn.Module):
    def __init__(self, radius=0.5) -> None:
        super().__init__()
        self.radius = radius

    def forward(self, x, r, cyl=None):
        # we want to pass the 
        e = r - x
        if cyl is not None:
            c_pos, c_vel = posVel2cyl(x, cyl, self.radius)
            return torch.hstack([e, c_pos, c_vel])
        else:
            return e

class processFig8TrajPolicyInput(torch.nn.Module):
    def __init__(self, use_error=True) -> None:
        super().__init__()
        self.use_error = use_error

    def forward(self, x, r, p):
        # we want to pass the 
        e = r - x
        if self.use_error is True:
            return torch.hstack([e,p])
        elif self.use_error is False:
            return torch.hstack([x,e,p])
        
class processP2PTrajPolicyInput(torch.nn.Module):
    def __init__(self, use_error=True) -> None:
        super().__init__()
        self.use_error = use_error

    def forward(self, x, r):
        # we want to pass the 
        e = r - x
        if self.use_error is True:
            return e
        elif self.use_error is False:
            return torch.hstack([x,e])

class radMultiplier(torch.nn.Module):
    def __init__(self, Ts, bs=1) -> None:
        super().__init__()
        self.Ts = Ts
        self.bs = bs

    def forward(self, i):
        # multiplier = 1 + idx * Ts * 0.5
        m = 1 + i * self.Ts * 0.01 # 0.15 # 0.5
        i = i + 1
        return i, m
    

class BimodalPolicy(torch.nn.Module):
    def __init__(self,
            insize=6+2,
            outsize=4, 
            bias=True,
            linear_map=torch.nn.Linear,
            nonlin=torch.nn.ReLU,
            hsizes=[20, 20, 20, 20]
        ) -> None:
        super().__init__()

        self.mode = []
        self.mode.append()
