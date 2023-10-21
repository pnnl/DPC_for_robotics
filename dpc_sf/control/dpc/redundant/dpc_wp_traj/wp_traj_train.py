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
from neuromancer.dynamics import integrators, ode

from dpc_sf.utils import pytorch_utils as ptu

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)

save_path = "policy/DPC/"
nstep = 100
epochs = 10 
iterations = 1
lr = 0.01
Ts = 0.1
minibatch_size = 10 # do autograd per minibatch_size
batch_size = 500 # 
dataset_mode = 'sinusoidal'

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

nx = 6 # state size
nr = 6 # reference size
nu = 3 # input size
nc = 2 # cylinder coordinates size

interp_u = lambda tq, t, u: u

class Dynamics(ode.ODESystem):
    def __init__(self, insize, outsize) -> None:
        super().__init__(insize=insize, outsize=outsize)
        self.f = lambda x, u: x @ A.T + u @ B.T
        self.in_features = insize
        self.out_features = outsize

    def ode_equations(self, xu):
        x = xu[:,0:6]
        u = xu[:,6:9]
        return self.f(x,u)

class Cat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args):
        # expects shape [bs, nx]
        return torch.hstack(args)
    
state_ref_cat = Cat()
state_ref_cat_node = Node(state_ref_cat, ['X', 'R'], ['XR'], name='cat')

# policy = Policy(bs=3333)

policy = blocks.MLP(
    insize=nx + nr, outsize=nu, bias=True,
    linear_map=torch.nn.Linear,
    nonlin=torch.nn.ReLU,
    hsizes=[20, 20, 20, 20]
)

policy_node = Node(policy, ['XR'], ['U'], name='policy')

state_input_cat = Cat()
state_input_cat_node = Node(state_input_cat, ['X', 'U'], ['XU'])

dynamics = Dynamics(insize=9, outsize=6)

integrator = integrators.Euler(dynamics, interp_u=interp_u, h=torch.tensor(Ts))

sys_node = Node(integrator, ['X', 'U'], ['X'], name='integrator')

cl_system = System([state_ref_cat_node, policy_node, sys_node])
# cl_system.show()

# Training dataset generation
x_range = 3.
r_range = 3.
cyl_range = 3.
end_pos = ptu.create_tensor([[[0.0,0.0,0.0,0.0,0.0,0.0]]])

def is_inside_cylinder(x, y, cx, cy, radius=0.5):
    """
    Check if a point (x,y) is inside a cylinder with center (cx, cy) and given radius.
    """
    distance = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return distance < radius

def generate_reference(nstep, nx, Ts, r_range, mode='linear'):
    """
    Generate a reference dataset.
    Parameters:
    - nstep: Number of steps
    - nx: Dimensionality of the reference
    - Ts: Time step
    - r_range: Range for random sampling
    - mode: 'linear' for straight line references, 'sinusoidal' for sinusoidal references
    """
    if mode == 'linear':
        start_point = r_range * torch.randn(1, 1, nx)
        end_point = r_range * torch.randn(1, 1, nx)

        pos_sample = []
        for dim in range(3):  # Only interpolate the positions (x, y, z)
            interpolated_values = torch.linspace(start_point[0, 0, dim], end_point[0, 0, dim], steps=nstep+1)
            pos_sample.append(interpolated_values)

        pos_sample = torch.stack(pos_sample, dim=-1)

    elif mode == 'sinusoidal':
        t = torch.linspace(0, nstep * Ts, nstep + 1)  # Generate time values
        
        freq = 1.0 / (nstep * Ts)  # Adjust frequency as needed
        amplitude = r_range / 2  # Adjust amplitude as needed

        # Random phase shift corresponding to a random start time in the interval [0, 2*pi]
        phase_shift = 2 * np.pi * torch.rand(1)

        pos_sample = amplitude * torch.sin(2 * np.pi * freq * t + phase_shift).unsqueeze(-1)
        pos_sample = torch.cat([pos_sample for _ in range(3)], dim=-1)  # Replicating for 3D (x, y, z)

    # Calculate the velocities
    vel_sample = (pos_sample[1:, :] - pos_sample[:-1, :]) / Ts
    # For the last velocity, we duplicate the last calculated velocity
    vel_sample = torch.cat([vel_sample, vel_sample[-1:, :]], dim=0)

    return pos_sample, vel_sample

def get_filtered_dataset(batch_size, nx, nstep, x_range, r_range, end_pos, cyl_range, mode='linear'): # mode = 'linear', 'sinusoidal'
    X = []
    R = []
    
    # Loop until the desired batch size is reached.
    print(f"generating dataset of batchsize: {batch_size}")
    while len(X) < batch_size:
        x_sample = x_range * torch.randn(1, 1, nx)

        pos_sample, vel_sample = generate_reference(nstep=nstep, nx=nx, Ts=Ts, r_range=r_range, mode=mode)

        # Rearrange to the desired order {x, xdot, y, ydot, z, zdot}
        r_sample = torch.zeros(1, nstep+1, nx)
        r_sample[0, :, 0] = pos_sample[:, 0]
        r_sample[0, :, 1] = vel_sample[:, 0]
        r_sample[0, :, 2] = pos_sample[:, 1]
        r_sample[0, :, 3] = vel_sample[:, 1]
        r_sample[0, :, 4] = pos_sample[:, 2]
        r_sample[0, :, 5] = vel_sample[:, 2]
        r_sample += end_pos
        
        X.append(x_sample)
        R.append(r_sample)
    
    # Convert lists to tensors.
    X = torch.cat(X, dim=0)
    R = torch.cat(R, dim=0)
    
    return {
        'X': X,
        'R': R,
    }

train_data = DictDataset(get_filtered_dataset(
    batch_size=batch_size,
    nx = nx,
    nstep = nstep,
    x_range = x_range,
    r_range = r_range,
    end_pos = end_pos,
    cyl_range = cyl_range,
    mode = dataset_mode
), name='train')

dev_data = DictDataset(get_filtered_dataset(
    batch_size=batch_size,
    nx = nx,
    nstep = nstep,
    x_range = x_range,
    r_range = r_range,
    end_pos = end_pos,
    cyl_range = cyl_range,
    mode = dataset_mode
), name='dev')

train_loader = torch.utils.data.DataLoader(train_data, batch_size=minibatch_size,
                                           collate_fn=train_data.collate_fn, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=minibatch_size,
                                         collate_fn=dev_data.collate_fn, shuffle=False)

r = variable('R')
u = variable('U')
x = variable('X')

action_loss = 0.1 * (u == 0.)^2  # control penalty
# regulation_loss = 5. * (x[:,:,::2] == r[:,:,::2])^2  # target position only, not velocity
regulation_loss = 5. * (x[:,:-1,:] == r[:,1:,:])^2

loss = PenaltyLoss([action_loss, regulation_loss], [])
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

    train_data = DictDataset(get_filtered_dataset(
        batch_size=batch_size,
        nx = nx,
        nstep = nstep,
        x_range = x_range,
        r_range = r_range,
        end_pos = end_pos,
        cyl_range = cyl_range,
    ), name='train')

    dev_data = DictDataset(get_filtered_dataset(
        batch_size=batch_size,
        nx = nx,
        nstep = nstep,
        x_range = x_range,
        r_range = r_range,
        end_pos = end_pos,
        cyl_range = cyl_range,
    ), name='dev')


    # apply new training data and learning rate to trainer
    trainer.train_data, trainer.dev_data, trainer.test_data = train_data, dev_data, dev_data
    optimizer.param_groups[0]['lr'] = lr


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

torch.save(policy_state_dict, save_path + "wp_traj.pth")

# del policy_state_dict



print('fin')

