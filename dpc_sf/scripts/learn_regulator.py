# reinstalling neuromancer 03_07_2023
# this script attempts to learn a regulator for the quad, the optimisation takes place
# but is insufficient to regulate with the given hyperparameters

import torch
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.plot import pltCL, pltPhase

from quad import Quadcopter

quad = Quadcopter()

# Node and System Class Setup
# ---------------------------

# closed loop system definition
mlp = blocks.MLP(17, 4, bias=True,
                 linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ReLU,
                 hsizes=[20, 20, 20, 20])
policy = Node(mlp, ['X'], ['U'], name='policy')

# xnext = lambda x, u: x @ A.T + u @ B.T
xnext = lambda x, u: x + quad.state_dot(x, u) * 0.1

# double_integrator = Node(xnext, ['X', 'U'], ['X'], name='integrator')
quad_node = Node(xnext, ['X', 'U'], ['X'], name='quad_node')

# cl_system = System([policy, double_integrator])
cl_system = System([policy, quad_node])

# Training Dataset Generation
# ---------------------------

def random_state(samples):
    pos_rand = torch.randn(samples, 1, 3) * 10
    quat_rand = torch.randn(samples, 1, 4) * torch.pi
    vel_rand = torch.randn(samples, 1, 3) * 5
    angv_rand = torch.randn(samples, 1, 3) * 0.3
    omegas_rand = torch.randn(samples, 1, 4) * 300 + 400
    state = torch.concatenate([pos_rand, quat_rand, vel_rand, angv_rand, omegas_rand], axis=2)
    return state

num_train_samples = 3333
num_dev_samples = 3333

train_data = DictDataset({'X': random_state(num_train_samples)}, name='train')
dev_data = DictDataset({'X': random_state(num_dev_samples)}, name='dev')

train_loader = torch.utils.data.DataLoader(train_data, batch_size=num_train_samples,
                                           collate_fn=train_data.collate_fn, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=num_dev_samples,
                                         collate_fn=dev_data.collate_fn, shuffle=False)

# Optimisation Problem
# --------------------

# Define optimization problem
u = variable('U')
x = variable('X')
action_loss = 0.001 * (u == 0.)^2  # control penalty
regulation_loss = 5. * (x == 0.)^2  # target position
loss = PenaltyLoss([action_loss, regulation_loss], [])
problem = Problem([cl_system], loss)
optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001)

# Optimise problem with a system rollout of 2 timesteps (relative degree)
# -----------------------------------------------------------------------

trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    dev_loader,
    optimizer,
    epochs=500,
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="test_loss",
    eval_metric='dev_loss',
    warmup=400,
)

# Train model with prediction horizon of 3
cl_system.nsteps = 3
best_model = trainer.train()

# Evaluate Best Model on a system rollout of 50 timesteps
# -------------------------------------------------------

# Test best model with prediction horizon of 50
problem.load_state_dict(best_model)
data = {'X': torch.ones(1, 1, 17, dtype=torch.float32)}
cl_system.nsteps = 50
trajectories = cl_system(data)
pltCL(Y=trajectories['X'][:,:,0:2].detach().reshape(51, 2), U=trajectories['U'][:,:,0:1].detach().reshape(50, 1), figname='cl.png')
pltPhase(X=trajectories['X'][:,:,0:2].detach().reshape(51, 2), figname='phase.png')

print('fin')