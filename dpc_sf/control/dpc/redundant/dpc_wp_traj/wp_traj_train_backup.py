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

save_path = "data/policy/DPC_traj/"
nstep = 100
epochs = 10 
iterations = 1
lr = 0.01
Ts = 0.1
minibatch_size = 10 # do autograd per minibatch_size
batch_size = 500 # 

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


class stateRefCat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, r, c):
        # expects shape [bs, nx]
        return torch.hstack([x,r,c])
    
class stateInputCat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, u):
        return torch.hstack([x,u])
    
state_ref_cat = stateRefCat()
state_ref_cat_node = Node(state_ref_cat, ['X', 'R', 'Cyl'], ['XRC'], name='cat')

# policy = Policy(bs=3333)

policy = blocks.MLP(
    insize=nx + nr + nc, outsize=nu, bias=True,
    linear_map=torch.nn.Linear,
    nonlin=torch.nn.ReLU,
    hsizes=[20, 20, 20, 20]
)

policy_node = Node(policy, ['XRC'], ['U'], name='policy')

state_input_cat = stateInputCat()
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

def get_filtered_dataset(batch_size, nx, nstep, x_range, r_range, end_pos, cyl_range):
    X = []
    R = []
    Cyl = []
    
    # Loop until the desired batch size is reached.
    print(f"generating dataset of batchsize: {batch_size}")
    while len(X) < batch_size:
        x_sample = x_range * torch.randn(1, 1, nx)

        # we sample the reference now as a linear interpolation between two random points
        # r_sample = torch.concatenate([r_range * torch.randn(1, 1, nx)] * (nstep + 1), dim=1) + end_pos
        start_point = r_range * torch.randn(1, 1, nx)
        end_point = r_range * torch.randn(1, 1, nx)
        
        pos_sample = []
        for dim in range(3):  # Only interpolate the positions (x, y, z)
            interpolated_values = torch.linspace(start_point[0, 0, dim], end_point[0, 0, dim], steps=nstep+1)
            pos_sample.append(interpolated_values)
        
        pos_sample = torch.stack(pos_sample, dim=-1)

        # Now, calculate the velocities
        vel_sample = (pos_sample[1:, :] - pos_sample[:-1, :]) / Ts
        # For the last velocity, we duplicate the last calculated velocity
        vel_sample = torch.cat([vel_sample, vel_sample[-1:, :]], dim=0)

        # Rearrange to the desired order {x, xdot, y, ydot, z, zdot}
        r_sample = torch.zeros(1, nstep+1, nx)
        r_sample[0, :, 0] = pos_sample[:, 0]
        r_sample[0, :, 1] = vel_sample[:, 0]
        r_sample[0, :, 2] = pos_sample[:, 1]
        r_sample[0, :, 3] = vel_sample[:, 1]
        r_sample[0, :, 4] = pos_sample[:, 2]
        r_sample[0, :, 5] = vel_sample[:, 2]
        r_sample += end_pos

        cyl_sample = torch.concatenate([cyl_range * torch.randn(1, 1, 2)] * (nstep + 1), dim=1)
        
        inside_cyl = False
        
        # Check if any state or reference point is inside the cylinder.
        for t in range(nstep + 1):
            if is_inside_cylinder(x_sample[0, 0, 0], x_sample[0, 0, 2], cyl_sample[0, t, 0], cyl_sample[0, t, 1]):
                inside_cyl = True
                break
            if is_inside_cylinder(r_sample[0, t, 0], r_sample[0, t, 2], cyl_sample[0, t, 0], cyl_sample[0, t, 1]):
                inside_cyl = True
                break
        
        if not inside_cyl:
            X.append(x_sample)
            R.append(r_sample)
            Cyl.append(cyl_sample)
    
    # Convert lists to tensors.
    X = torch.cat(X, dim=0)
    R = torch.cat(R, dim=0)
    Cyl = torch.cat(Cyl, dim=0)
    
    return {
        'X': X,
        'R': R,
        'Cyl': Cyl
    }

def validate_dataset(dataset):
    X = dataset['X']
    R = dataset['R']
    Cyl = dataset['Cyl']
    
    batch_size = X.shape[0]
    nstep = R.shape[1] - 1

    print("validating dataset...")
    for i in range(batch_size):
        for t in range(nstep + 1):
            # Check initial state.
            if is_inside_cylinder(X[i, 0, 0], X[i, 0, 2], Cyl[i, t, 0], Cyl[i, t, 1]):
                return False, f"Initial state at index {i} lies inside the cylinder."
            # Check each reference point.
            if is_inside_cylinder(R[i, t, 0], R[i, t, 2], Cyl[i, t, 0], Cyl[i, t, 1]):
                return False, f"Reference at time {t} for batch index {i} lies inside the cylinder."

    return True, "All points are outside the cylinder."

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

validate_dataset(train_data.datadict)
validate_dataset(dev_data.datadict)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=minibatch_size,
                                           collate_fn=train_data.collate_fn, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=minibatch_size,
                                         collate_fn=dev_data.collate_fn, shuffle=False)

# ts = variable('Ts')
cyl = variable('Cyl')
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

    validate_dataset(train_data.datadict)
    validate_dataset(dev_data.datadict)

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

