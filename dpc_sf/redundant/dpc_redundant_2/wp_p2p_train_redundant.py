"""
description:

This DPC tracks a reference, and avoids a cylinder of random position, but fixed
size. 
"""

import torch 

from neuromancer.system import Node, System
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import BarrierLoss
from neuromancer.modules import blocks
from neuromancer.plot import pltCL
from neuromancer.dynamics import integrators, ode

from dpc_sf.utils import pytorch_utils as ptu

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)

# Hyperparameters
# ---------------

radius = 0.5 # 0.5
save_path = "policy/DPC/"
nstep = 100
epochs = 10 
iterations = 3
lr = 0.01
Ts = 0.1
minibatch_size = 10 # do autograd per minibatch_size
batch_size = 500 # 
x_range = 3. # 3.
r_range = 3. # 3.
cyl_range = 3. # 3.
use_integrator = True
use_rad_multiplier = False
train = True

# Dynamics Definition
# -------------------

nx = 6 # state size
nr = 6 # reference size
nu = 3 # input size
nc = 2 # cylinder coordinates size

interp_u = lambda tq, t, u: u

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


# System Definition
# -----------------


# variables
cyl = variable('Cyl')
r = variable('R')
u = variable('U')
x = variable('X')
idx = variable('Idx')
m = variable('M') # radius multiplier, increase radius into horizon future
I_err = variable('I_err')

# constraints

# THE WORKING CONSTRAINT
# multiplier = 1 # + idx * Ts * 0.5
# Q_con = 1000000.
# cylinder_constraint = Q_con * ((radius**2 * multiplier <= (x[:,:,0]-cyl[:,:,0])**2 + (x[:,:,2]-cyl[:,:,1])**2)) ^ 2

# EXPERIMENTAL CONSTRAINT
Q_con = 1000000.
cylinder_constraint = Q_con * ((radius**2 * m[:,:,0] <= (x[:,:,0]-cyl[:,:,0])**2 + (x[:,:,2]-cyl[:,:,1])**2)) ^ 2


# Classes:
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


class IntegratorPolicy(torch.nn.Module):
    def __init__(
            self, 
            init_gain, 
            bs=1,
            insize=nx + nr + nc + 3, 
            outsize=nu, 
            bias=True,
            linear_map=torch.nn.Linear,
            nonlin=torch.nn.ReLU,
            hsizes=[20, 20, 20, 20]
        ) -> None:
        super().__init__()
        self.gain = init_gain
        self.mlp = blocks.MLP(
            insize=insize, outsize=outsize, bias=bias,
            linear_map=linear_map,
            nonlin=nonlin,
            hsizes=hsizes
        )

        # anti windup minimum and maximum integration error
        self.int_err_min = ptu.create_zeros([bs, 3])
        self.int_err_min += ptu.create_tensor([[-50, -50, -50]])

        self.int_err_max = ptu.create_zeros([bs, 3])
        self.int_err_max += ptu.create_tensor([[50, 50, 50]])

    def forward(self, xrc, i_err):

        xrci = torch.hstack([xrc, i_err])

        mlp_u = self.mlp(xrci)

        i_u = self.gain * i_err
        u = mlp_u + i_u

        x = xrc[:,0:6]
        r = xrc[:,6:12]
        # c = xrci[:,12:14]
        # i = xrci[:,14:17]
        err = r - x
        pos_err = err[:,::2]
        i_err  = i_err + pos_err

        # Detach the tensor from the graph.
        # self.int_error = self.int_error.detach()

        # anti windup clip the integration error
        i_err = torch.clip(i_err, min = self.int_err_min, max = self.int_err_max)

        # print(i_u)
        return u, i_err


class stateRefCat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, r, c):
        # expects shape [bs, nx]
        return torch.hstack([x,r,c])

class Cat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args):
        # expects shape [bs, nx]
        return torch.hstack(args)

class radMultiplier(torch.nn.Module):
    def __init__(self, Ts, bs=1) -> None:
        super().__init__()
        self.Ts = Ts
        self.bs = bs

    def forward(self, i):
        # multiplier = 1 + idx * Ts * 0.5
        m = 1 + i * self.Ts * 0.05 # 0.15 # 0.5
        # m = ptu.create_tensor(m).unsqueeze(0).unsqueeze(0)
        # m = torch.vstack([m]*self.bs)
        i = i + 1
        # print(f'm is: {m}')
        # print(f'i is: {i}')
        return i, m


# Nodes:
node_list = []

if use_rad_multiplier:
    rad_multiplier = radMultiplier(Ts=Ts, bs=batch_size)
    rad_multiplier_node = Node(rad_multiplier, ['Idx'], ['Idx', 'M'], name='rad_multiplier')
    node_list.append(rad_multiplier_node)

state_ref_cat = stateRefCat()
state_ref_cat_node = Node(state_ref_cat, ['X', 'R', 'Cyl'], ['XRC'], name='cat')
node_list.append(state_ref_cat_node)

if use_integrator is True:
    
    policy = IntegratorPolicy(
        init_gain=ptu.create_tensor([0.1,0.1,0.1]),
        insize=nx + nr + nc + 3, outsize=nu, bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ReLU,
        hsizes=[20, 20, 20, 20]
    )
    policy_node = Node(policy, ['XRC', 'I_err'], ['U', 'I_err'], name='policy')
    node_list.append(policy_node)

elif use_integrator is False:
    policy = blocks.MLP(
        insize=nx + nr + nc, outsize=nu, bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ReLU,
        hsizes=[20, 20, 20, 20]
    )
    policy_node = Node(policy, ['XRC'], ['U'], name='policy')
    node_list.append(policy_node)
else:
    print("invalid integrator choice detected")

dynamics = Dynamics(insize=9, outsize=6)
integrator = integrators.Euler(dynamics, interp_u=interp_u, h=torch.tensor(Ts))
dynamics_node = Node(integrator, ['X', 'U'], ['X'], name='dynamics')
node_list.append(dynamics_node)

print(f'node list used in cl_system: {node_list}')
cl_system = System(node_list)
# cl_system = System([rad_multiplier_node, state_ref_cat_node, policy_node, dynamics_node])
# cl_system.show()

# Training dataset generation
# ---------------------------

"""
Algorithm for dataset generation:

- we randomly sample 3 points for each rollout, quad start state, quad reference state, cylinder position
- if a particular datapoint has a start or reference state that is contained within the random cylinder 
  we discard that datapoint and try again. The output of this is a "filtered" dataset in "get_filtered_dataset"
- I also have a validation function to check the filtered datasets produced here, but you can ignore that
- these filtered datasets are then wrapped in DictDatasets and then wrapped in torch dataloaders

NOTE:
- the minibatch size used in the torch dataloader can play a key role in reducing the final steady state error
  of the system
    - If the minibatch is too large we will not get minimal steady state error, minibatch of 10 has proven good
"""

# average end position, for if we want to go to a certain location more than others, 0. if not
end_pos = ptu.create_tensor([[[0.0,0.0,0.0,0.0,0.0,0.0]]])

def is_inside_cylinder(x, y, cx, cy, radius=radius):
    """
    Check if a point (x,y) is inside a cylinder with center (cx, cy) and given radius.
    """
    distance = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return distance < radius

def get_filtered_dataset(batch_size, nx, nstep, x_range, r_range, end_pos, cyl_range):
    """
    Generate a filtered dataset of samples where none of the state or reference points are inside a cylinder.

    Parameters:
    - batch_size (int): Desired number of data samples in the batch.
    - nx (int): Dimension of the state sample.
    - nstep (int): Number of steps for reference and cylinder data.
    - x_range (float): Scaling factor for state data.
    - r_range (float): Scaling factor for reference data.
    - end_pos (float): Constant offset for reference data.
    - cyl_range (float): Scaling factor for cylinder data.

    Returns:
    - dict: A dictionary containing:
        'X': Tensor of state data of shape [batch_size, 1, nx].
        'R': Tensor of reference data of shape [batch_size, nstep + 1, nx].
        'Cyl': Tensor of cylinder data of shape [batch_size, nstep + 1, 2].
        'Idx': Zero tensor indicating starting index for each sample in batch.
        'M': Tensor with ones indicating starting multiplier for each sample.
        'I_err': Zero tensor indicating initial error for each sample in batch.

    """
    X = []
    R = []
    Cyl = []
    
    # Loop until the desired batch size is reached.
    print(f"generating dataset of batchsize: {batch_size}")
    while len(X) < batch_size:
        x_sample = x_range * torch.randn(1, 1, nx)
        r_sample = torch.concatenate([r_range * torch.randn(1, 1, nx)] * (nstep + 1), dim=1) + end_pos
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
        'Cyl': Cyl,
        'Idx': ptu.create_zeros([batch_size,1,1]), # start idx
        'M': torch.ones([batch_size, 1, 1]), # start multiplier
        'I_err': ptu.create_zeros([batch_size,1,3])
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

# Optimization Problem Setup
# --------------------------

"""
NOTE:
- Use of a BarrierLoss, which provides a loss which increases the closer to a constraint violation
  we come, has proven much more effective than a PenaltyLoss in obstacle avoidance. The PenaltyLoss 
  is either on or off, and so has a poor learning signal for differentiable problems, but the BarrierLoss
  provides a constant learning signal which has proven effective.
- Not penalising velocity, and only position has frequently led to controllers that do not converge
  to zero velocity, and therefore drift over time. Therefore velocities have been kept in the 
  regulation loss term.
"""

# loss function
action_loss = 0.1 * (u == 0.)^2  # control penalty
# regulation_loss = 5. * (x[:,:,::2] == r[:,:,::2])^2  # target position only, not velocity
regulation_loss = 5. * (x[:,:,:] == r[:,:,:])^2
# loss = PenaltyLoss([action_loss, regulation_loss], [cylinder_constraint])
loss = BarrierLoss([action_loss, regulation_loss], [cylinder_constraint])

problem = Problem([cl_system], loss, grad_inference=True)
optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

# Training
# --------

"""
NOTE:
- Multiple iterations with new datasets at lower learning rates has proven very effective at generating
  superior policies
- 
"""
if train is True:
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
else:
    best_model = torch.load("policy/DPC/wp_p2p.pth")

# Test Inference
# --------------
problem.load_state_dict(best_model)
data = {
    'X': torch.zeros(1, 1, nx, dtype=torch.float32), 
    'R': torch.concatenate([torch.tensor([[[2, 0, 2, 0, 2, 0]]])]*(nstep+1), dim=1), 
    'Cyl': torch.concatenate([torch.tensor([[[1,1]]])]*(nstep+1), dim=1), 
    'Idx': torch.vstack([torch.tensor([0.0])]).unsqueeze(1),
    'M': torch.ones([1, 1, 1]), # start multiplier
    'I_err': ptu.create_zeros([1,1,3])
}
cl_system.nsteps = nstep
print(f"testing model over {nstep} timesteps...")
trajectories = cl_system(data)
pltCL(
    Y=trajectories['X'].detach().reshape(nstep + 1, 6), 
    U=trajectories['U'].detach().reshape(nstep, 3), 
    figname=f'cl.png'
)

# Save the MLP parameters
# -----------------------

policy_state_dict = {}
for key, value in best_model.items():
    if "callable." in key:
        new_key = key.split("nodes.0.nodes.1.")[-1]
        policy_state_dict[new_key] = value

torch.save(policy_state_dict, save_path + "wp_p2p.pth")

print('fin')

