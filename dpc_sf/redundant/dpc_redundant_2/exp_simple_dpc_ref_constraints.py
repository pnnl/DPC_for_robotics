"""
description:

this learned dpc should be slightly different to exp_simple
it will track a reference rather than have an error regulated to 0
"""

import numpy as np
import torch 
from tqdm import tqdm
import copy

from neuromancer.system import Node, System
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.plot import pltCL, pltPhase
from neuromancer.dynamics import integrators

from dpc_sf.utils import pytorch_utils as ptu

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)

save_path = "policy/DPC/"
nstep = 200
epochs = 60 
iterations = 2
lr = 0.01

Ts = 0.1
Ts_end = 0.1
Ts_multiplier = (Ts_end/Ts) ** (1/nstep)

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

# B *= 0.1

nx = 6
nu = 3

dx = lambda x, u: x @ A.T + u @ B.T

# TODO change the integrator to be torchdiffEq
# dx = blocks.MLP(sys.nx + sys.nu, sys.nx, bias=True, linear_map=torch.nn.Linear, nonlin=torch.nn.ELU,
#               hsizes=[20 for h in range(3)])
interp_u = lambda tq, t, u: u
# integrator = integrators.Euler(dx, h=torch.tensor(0.1), interp_u=interp_u)
# system_node = Node(integrator, ['xn', 'U'], ['xn'])

# xnext = lambda x, u, Ts=Ts: x + f(x, u) * Ts

class Dynamics(torch.nn.Module):
    def __init__(self, Ts_multiplier=1.2) -> None:
        super().__init__()
        self.Ts_multiplier = Ts_multiplier

    def forward(self, x, u, Ts):
        xnext = x + dx(x, u) * Ts
        Ts = Ts * self.Ts_multiplier
        Ts = torch.clip(Ts, -1.0, 1.0)
        # print(Ts)
        return xnext, Ts

# class Policy(torch.nn.Module):
#     def __init__(self, bs=1) -> None:
#         super().__init__()
# 
#         # the nonlinear proportional gain to be learned by a NN
#         self.p_gain = blocks.MLP(
#             insize=nx * 2, outsize=nu, bias=True,
#             linear_map=torch.nn.Linear,
#             nonlin=torch.nn.ReLU,
#             hsizes=[20, 20, 20, 20]
#         )
# 
#         # the integral term gain to be learned
#         # self.i_gain = torch.nn.Parameter(ptu.create_tensor([0.0001,0.0001,0.0001]))
#         # self.i_gain_multiplier = torch.nn.Parameter(ptu.create_tensor([1.0,1.0,1.0]))
#         self.i_gain = ptu.create_tensor([0.0,0.0,0.0])#  * self.i_gain_multiplier
# 
#         # the integrated position error
#         self.i_pos_err = ptu.create_zeros([bs, 3])
# 
#     def reset_integral(self, bs=1):
#         self.i_pos_err = ptu.create_zeros([bs, 3])
# 
#     def forward(self, xr):
#         
#         # accumulate integral error
#         positions = xr[:,::2]
#         x_pos = positions[:,:3]
#         r_pos = positions[:,3:]
#         self.i_pos_err = self.i_pos_err - x_pos + r_pos
# 
#         # illegal due to in place operation on two gradient tensors
#         # u = self.p_gain(xr) + self.i_gain * self.i_pos_err
# 
#         u_p = self.p_gain(xr)
#         u_i = self.i_gain * self.i_pos_err
#         u_out = u_p + u_i
# 
#         return u_out


class stateRefCat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, r, c, t):
        # expects shape [bs, nx]
        return torch.hstack([x,r,c,t])
    
state_ref_cat = stateRefCat()

state_ref_cat_node = Node(state_ref_cat, ['X', 'R', 'Cyl', 'Ts'], ['XRCT'], name='cat')

# policy = Policy(bs=3333)

policy = blocks.MLP(
    insize=nx * 2 + 2 + 1, outsize=nu, bias=True,
    linear_map=torch.nn.Linear,
    nonlin=torch.nn.ReLU,
    hsizes=[20, 20, 20, 20]
)

policy_node = Node(policy, ['XRCT'], ['U'], name='policy')

sys = Dynamics(Ts_multiplier=Ts_multiplier)

sys_node = Node(sys, ['X', 'U', 'Ts'], ['X', 'Ts'], name='integrator')

cl_system = System([state_ref_cat_node, policy_node, sys_node])
# cl_system.show()

# Training dataset generation
# train_range = 3. # 3.
x_range = 3.
r_range = 3.
cyl_range = 3.
end_pos = ptu.create_tensor([[[0.0,0.0,0.0,0.0,0.0,0.0]]])
train_data = DictDataset({
    'X': x_range*torch.randn(3333, 1, nx), 
    'R': torch.concatenate([r_range*torch.randn(3333, 1, nx)] * (nstep + 1), dim=1) + end_pos,
    'Cyl': torch.concatenate([cyl_range*torch.randn(3333, 1, 2)] * (nstep + 1), dim=1),
    'Ts': torch.vstack([torch.tensor([Ts])] * 3333).unsqueeze(1)
}, name='train')  # Split conditions into train and dev
dev_data = DictDataset({
    'X': x_range*torch.randn(3333, 1, nx), 
    'R': torch.concatenate([r_range*torch.randn(3333, 1, nx)] * (nstep + 1), dim=1) + end_pos,
    'Cyl': torch.concatenate([cyl_range*torch.randn(3333, 1, 2)] * (nstep + 1), dim=1),
    'Ts': torch.vstack([torch.tensor([Ts])] * 3333).unsqueeze(1)
}, name='dev')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=3333,
                                           collate_fn=train_data.collate_fn, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=3333,
                                         collate_fn=dev_data.collate_fn, shuffle=False)

ts = variable('Ts')
cyl = variable('Cyl')
r = variable('R')
u = variable('U')
x = variable('X')

action_loss = 0.1 * (u == 0.)^2  # control penalty
# regulation_loss = 5. * (x[:,:,::2] == r[:,:,::2])^2  # target position only, not velocity
regulation_loss = 5. * (x[:,:,:] == r[:,:,:])^2

def is_in_cylinder(self, X, Y, multiplier):
    return self.radius ** 2 * multiplier <= (X - self.x_pos)**2 + (Y - self.y_pos)**2

radius = 0.5
# x_pos = 1
# y_pos = 1

# where k is the timestep
# multiplier = 1 + k * Ts * 0.5
multiplier = 1
Q_con = 1000000.
cylinder_constraint = Q_con * ((radius**2 * multiplier <= (x[:,:,0]-cyl[:,:,0])**2 + (x[:,:,2]-cyl[:,:,1])**2)) ^ 2

loss = PenaltyLoss([action_loss, regulation_loss], [cylinder_constraint])
problem = Problem([cl_system], loss)
optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

# Train model with prediction horizon of train_nsteps
for i in range(iterations):
    print(f'training with prediction horizon: {nstep}, lr: {lr}')

    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        dev_loader,
        optimizer,
        epochs=epochs,
        patience=epochs,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric='dev_loss',
        warmup=400,
    )

    cl_system.nsteps = nstep
    best_model = trainer.train()
    trainer.model.load_state_dict(best_model)
    lr /= 5.0
    # train_nsteps *= 2

    # update the prediction horizon
    cl_system.nsteps = nstep

    # get another set of data
    x_range = 1.
    r_range = 1.
    cyl_range = 0.2
    end_pos = ptu.create_tensor([[[2.0,0.0,2.0,0.0,2.0,0.0]]])
    train_data = DictDataset({
        'X': x_range*torch.randn(3333, 1, nx), 
        'R': torch.concatenate([r_range*torch.randn(3333, 1, nx)] * (nstep + 1), dim=1) + end_pos,
        'Cyl': torch.concatenate([cyl_range*torch.randn(3333, 1, 2)] * (nstep + 1), dim=1),
        'Ts': torch.vstack([torch.tensor([Ts])] * 3333).unsqueeze(1)
    }, name='train')  # Split conditions into train and dev
    dev_data = DictDataset({
        'X': x_range*torch.randn(3333, 1, nx), 
        'R': torch.concatenate([r_range*torch.randn(3333, 1, nx)] * (nstep + 1), dim=1) + end_pos,
        'Cyl': torch.concatenate([cyl_range*torch.randn(3333, 1, 2)] * (nstep + 1), dim=1),
        'Ts': torch.vstack([torch.tensor([Ts])] * 3333).unsqueeze(1)
    }, name='dev')

    # apply new training data and learning rate to trainer
    trainer.train_data, trainer.dev_data, trainer.test_data = train_data, dev_data, dev_data
    optimizer.param_groups[0]['lr'] = lr

# trainer = Trainer(
#     problem,
#     train_loader,
#     dev_loader,
#     dev_loader,
#     optimizer,
#     epochs=epochs,
#     train_metric="train_loss",
#     dev_metric="dev_loss",
#     test_metric="test_loss",
#     eval_metric='dev_loss',
#     warmup=400,
# )
# 
# # Train model with prediction horizon of 2
# cl_system.nsteps = nstep
# best_model = trainer.train()

problem.load_state_dict(best_model)

data = {
    'X': torch.zeros(1, 1, nx, dtype=torch.float32), 
    'R': torch.concatenate([torch.tensor([[[2, 0, 2, 0, 2, 0]]])]*(nstep+1), dim=1), 
    'Cyl': torch.concatenate([torch.tensor([[[1,1]]])]*(nstep+1), dim=1), 
    'Ts': torch.vstack([torch.tensor([Ts])]).unsqueeze(1)}
cl_system.nsteps = nstep
print(f"testing model over {nstep} timesteps...")
#cl_system.nodes[1].callable.reset_integral(bs=1)
trajectories = cl_system(data)
pltCL(Y=trajectories['X'].detach().reshape(nstep + 1, 6), U=trajectories['U'].detach().reshape(nstep, 3), figname='cl.png')
# pltPhase(X=trajectories['X'].detach().reshape(51, 6), figname='phase.png')

# save the MLP parameters:
# Extract required data
policy_state_dict = {}
for key, value in best_model.items():
    if "callable." in key:
        new_key = key.split("nodes.0.nodes.1.")[-1]
        policy_state_dict[new_key] = value

torch.save(policy_state_dict, save_path + "policy_state_dict_ref_constraints.pth")

# del policy_state_dict



print('fin')

