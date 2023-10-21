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

torch.random_seed(0)

save_path = "policy/DPC/"
nstep = 200
epochs = 60
iterations = 1
lr = 0.01

Ts = 0.1

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

f = lambda x, u: x @ A.T + u @ B.T

# TODO change the integrator to be torchdiffEq
# dx = blocks.MLP(sys.nx + sys.nu, sys.nx, bias=True, linear_map=torch.nn.Linear, nonlin=torch.nn.ELU,
#               hsizes=[20 for h in range(3)])
# interp_u = lambda tq, t, u: u
# integrator = integrators.Euler(dx, h=torch.tensor(0.1), interp_u=interp_u)
# system_node = Node(integrator, ['xn', 'U'], ['xn'])

xnext = lambda x, u: x + f(x, u) * Ts

mlp = blocks.MLP(insize=nx * 2, outsize=nu, bias=True,
                 linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ReLU,
                 hsizes=[20, 20, 20, 20])

class stateRefCat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, r):
        # expects shape [bs, nx]
        return torch.hstack([x,r])
    
state_ref_cat = stateRefCat()

state_ref_cat_node = Node(state_ref_cat, ['X', 'R'], ['XR'])

policy = Node(mlp, ['XR'], ['U'], name='policy')

sys = Node(xnext, ['X', 'U'], ['X'], name='integrator')

cl_system = System([state_ref_cat_node, policy, sys])
# cl_system.show()

# Training dataset generation
train_range = 3. # 3.
train_data = DictDataset({'X': train_range*torch.randn(3333, 1, nx), 'R': torch.concatenate([train_range*torch.randn(3333, 1, nx)] * (nstep + 1), dim=1)}, name='train')  # Split conditions into train and dev
dev_data = DictDataset({'X': train_range*torch.randn(3333, 1, nx), 'R': torch.concatenate([train_range*torch.randn(3333, 1, nx)] * (nstep + 1), dim=1)}, name='dev')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=3333,
                                           collate_fn=train_data.collate_fn, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=3333,
                                         collate_fn=dev_data.collate_fn, shuffle=False)


r = variable('R')
u = variable('U')
x = variable('X')
action_loss = 0.1 * (u == 0.)^2  # control penalty
# regulation_loss = 5. * (x[:,:,::2] == r[:,:,::2])^2  # target position only, not velocity
regulation_loss = 5. * (x[:,1:,:] == r[:,1:,:])^2
loss = PenaltyLoss([action_loss, regulation_loss], [])
problem = Problem([cl_system], loss)
optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)

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
    lr /= 2.0
    # train_nsteps *= 2

    # update the prediction horizon
    cl_system.nsteps = nstep

    # get another set of data
    train_data = DictDataset({'X': train_range*torch.randn(3333, 1, nx), 'R': torch.concatenate([train_range*torch.randn(3333, 1, nx)] * (nstep + 1), dim=1)}, name='train')  # Split conditions into train and dev
    dev_data = DictDataset({'X': train_range*torch.randn(3333, 1, nx), 'R': torch.concatenate([train_range*torch.randn(3333, 1, nx)] * (nstep + 1), dim=1)}, name='dev')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=3333,
                                            collate_fn=train_data.collate_fn, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=3333,
                                            collate_fn=dev_data.collate_fn, shuffle=False)

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



data = {'X': torch.ones(1, 1, nx, dtype=torch.float32), 'R': torch.concatenate([torch.tensor([[[-1, 0, -1, 0, -1, 0]]])]*(nstep+1), dim=1)}
cl_system.nsteps = nstep
print(f"testing model over {nstep} timesteps...")
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

torch.save(policy_state_dict, save_path + "policy_state_dict_ref.pth")

# del policy_state_dict



print('fin')

