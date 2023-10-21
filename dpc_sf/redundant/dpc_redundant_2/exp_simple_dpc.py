"""
In this experiment I will replace the quadcopters DPC with 3 x double
integrators in 3 dimensions. i.e. have a ball in space with no gravity 
that can accelerate in any direction, and see if we can train that. 

This dpc WORKS and is as if we are providing thrust_sp rather than vel_sp
meaning that we could simplify the 
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

save_path = "policy/DPC/"

# double integrator example:
# A = torch.tensor([[0.0, 1.0],
#                 [0.0, 0.0]])
# B = torch.tensor([[0.0],
#                 [1.0]])

Ts = 0.1

# A = torch.block_diag(A,A,A)
# B = torch.block_diag(B,B,B)

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

mlp = blocks.MLP(nx, nu, bias=True,
                 linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ReLU,
                 hsizes=[20, 20, 20, 20])

policy = Node(mlp, ['X'], ['U'], name='policy')

sys = Node(xnext, ['X', 'U'], ['X'], name='integrator')

cl_system = System([policy, sys])
# cl_system.show()

# Training dataset generation
range = 3. # 3.
train_data = DictDataset({'X': range*torch.randn(3333, 1, nx)}, name='train')  # Split conditions into train and dev
dev_data = DictDataset({'X': range*torch.randn(3333, 1, nx)}, name='dev')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=3333,
                                           collate_fn=train_data.collate_fn, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=3333,
                                         collate_fn=dev_data.collate_fn, shuffle=False)


u = variable('U')
x = variable('X')
action_loss = 0.1 * (u == 0.)^2  # control penalty
regulation_loss = 5. * (x == 0.)^2  # target position
loss = PenaltyLoss([action_loss, regulation_loss], [])
problem = Problem([cl_system], loss)
optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)

trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    dev_loader,
    optimizer,
    epochs=60,
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="test_loss",
    eval_metric='dev_loss',
    warmup=400,
)

# Train model with prediction horizon of 2
cl_system.nsteps = 100
best_model = trainer.train()

problem.load_state_dict(best_model)
data = {'X': torch.ones(1, 1, nx, dtype=torch.float32)}
cl_system.nsteps = 100
print(f"testing model over {100} timesteps...")
trajectories = cl_system(data)
pltCL(Y=trajectories['X'].detach().reshape(101, 6), U=trajectories['U'].detach().reshape(100, 3), figname='cl.png')
# pltPhase(X=trajectories['X'].detach().reshape(51, 6), figname='phase.png')

# save the MLP parameters:
# Extract required data
policy_state_dict = {}
for key, value in best_model.items():
    if "callable." in key:
        new_key = key.split("nodes.0.nodes.0.")[-1]
        policy_state_dict[new_key] = value

torch.save(policy_state_dict, save_path + "policy_state_dict.pth")

# del policy_state_dict



print('fin')

