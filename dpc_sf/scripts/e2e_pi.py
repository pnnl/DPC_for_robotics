from dpc_sf.control.pi.pi import XYZ_Vel
from dpc_sf.dynamics.eom_pt import state_dot_nm
from tqdm import tqdm
import numpy as np
from dpc_sf.utils.animation import Animator
import copy
import matplotlib.pyplot as plt

import torch

import dpc_sf.utils.pytorch_utils as ptu

from dpc_sf.dynamics.eom_pt import state_dot_nm
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.control.trajectory.trajectory import waypoint_reference

class Visualiser():
    def __init__(self, reference=waypoint_reference('wp_p2p', average_vel=1.6)) -> None:
        self.reference=reference
        self.x = []
        self.u = []
        self.r = []
        self.t = []

    def save(self, x, u, r, t):
        self.x.append(copy.deepcopy(x))
        self.u.append(np.copy(u))
        self.r.append(np.copy(r))
        self.t.append(t)

    def animate(self, x_p=None, drawCylinder=False):
        animator = Animator(
            states=np.vstack(self.x), 
            times=np.array(self.t), 
            reference_history=np.vstack(self.r), 
            reference=self.reference, 
            reference_type=self.reference.type, 
            drawCylinder=drawCylinder,
            state_prediction=x_p
        )
        animator.animate() # does not contain plt.show()      

vis = Visualiser()

print("testing the classical control system from the github repo")
Ts = 0.01
ctrl = XYZ_Vel(Ts=Ts)
R = waypoint_reference('wp_p2p', average_vel=1.6)
state = quad_params["default_init_state_pt"][0:13]
batch_size = 100

# initial conditions 2D
x = torch.vstack([state]*batch_size)

# need convert to NED
x[:,2] *= -1
x[:,9] *= -1

for i in tqdm(range(400)):
    t = i*Ts
    r = ptu.from_numpy(R(t))
    vel_sp = torch.vstack([torch.cos(torch.tensor([t*1, t*1, t*2])) * 5] * batch_size)

    u = ctrl(x, vel_sp)

    x[0,2] *= -1
    x[0,9] *= -1
    r[2] *= -1
    r[9] *= -1

    if i % 20 == 0:
        if x[0,:].isnan().any() or u[0,:].isnan().any():
            print(f"x: {x[0,:]}")
            print(f"u: {u[0,:]}")
            vis.animate()
            plt.show()   

        vis.save(x[0,:],u[0,:],r,t)

    x += state_dot_nm(x, u, params=quad_params, include_actuators=False) * Ts
    print(f"x: {x[0,:]}")

    x[0,2] *= -1
    x[0,9] *= -1
    r[2] *= -1
    r[9] *= -1

vis.animate()
plt.show()

print('fin')