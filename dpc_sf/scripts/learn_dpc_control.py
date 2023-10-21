
import torch
import torch.nn as nn
import neuromancer.slim as slim
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
from casadi import *
import casadi

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.constraint import Variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss, BarrierLoss, AugmentedLagrangeLoss
from neuromancer.modules import blocks
from neuromancer.system import Node
import neuromancer.arg as arg

from env import Sim
from quad import Quadcopter
from control.trajectory import waypoint_reference, equation_reference
import utils.pytorch_utils as ptu

# Simulation Environment
# ----------------------

dt = 0.1
Ti = 0
Tf = 4
reference_type = 'wp_p2p' # 'fig8', 'wp_traj', 'wp_p2p'
backend = 'eom' # 'eom', 'mj'
integrator_type = 'euler' # 'euler', 'RK4'

quad = Quadcopter()

state = np.array([
    0,                  # x
    0,                  # y
    0,                  # z
    1,                  # q0
    0,                  # q1
    0,                  # q2
    0,                  # q3
    0,                  # xdot
    0,                  # ydot
    0,                  # zdot
    0,                  # p
    0,                  # q
    0,                  # r
    522.9847140714692,  # wM1
    522.9847140714692,  # wM2
    522.9847140714692,  # wM3
    522.9847140714692   # wM4
])

if backend == 'mj':
    pass
elif backend == 'eom':
    state = ptu.from_numpy(state)

# setup trajectory
if reference_type == 'wp_traj':
    reference = waypoint_reference(type=reference_type, average_vel=1.6)
elif reference_type == 'wp_p2p':
    reference = waypoint_reference(type=reference_type, average_vel=0.5)
elif reference_type == 'fig8':
    reference = equation_reference(type=reference_type, average_vel=0.6)

env = Sim(
    ### parameters for both backends
    dt=dt,
    Ti=Ti,
    Tf=Tf,
    params=quad.params,
    backend=backend,
    init_state=state,
    reference=reference,
    integrator_type=integrator_type,
    ### eom specific arguments
    state_dot=quad.state_dot,
)

# Dataset
# -------

data_seed = 408  # random seed used for simulated data
np.random.seed(data_seed)
torch.manual_seed(data_seed)

nsim = 5000  # number of datapoints: increase sample density for more robust results
# create dictionaries with sampled datapoints with uniform distribution
samples_train = {"p": env.uniform_random_state(nsim)}
samples_dev = {"p": env.uniform_random_state(nsim)}
samples_test = {"p": env.uniform_random_state(nsim)}

# create named dictionary datasets
train_data = DictDataset(samples_train, name='train')
dev_data = DictDataset(samples_dev, name='dev')
test_data = DictDataset(samples_test, name='test')

# create torch dataloaders for the Trainer
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=0,
                                           collate_fn=train_data.collate_fn, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=32, num_workers=0,
                                         collate_fn=dev_data.collate_fn, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=0,
                                         collate_fn=test_data.collate_fn, shuffle=True)

# Policy Definition
# -----------------

n_hidden = 80                    # Number of hidden states of the solution map
n_layers = 4                     # Number of hidden layers of the solution map

# define neural architecture for the solution map
func = blocks.MLP(insize=17, outsize=4,
                linear_map=slim.maps['linear'],
                nonlin=nn.ReLU,
                hsizes=[n_hidden] * n_layers )

# define symbolic solution map with concatenated features (problem parameters)
sol_map = Node(func, ['p'], ['x'], name='map')

# Objective and Constraints in NM
# -------------------------------

"""
variable is a basic symbolic abstraction in Neuromancer
   x = variable("variable_name")                      (instantiates new variable)  
variable construction supports:
   algebraic expressions:     x**2 + x**3 + 5     (instantiates new variable)  
   slicing:                   x[:, i]             (instantiates new variable)  
   pytorch callables:         torch.sin(x)        (instantiates new variable)  
   constraints definition:    x <= 1.0            (instantiates Constraint object) 
   objective definition:      x.minimize()        (instantiates Objective object) 
to visualize computational graph of the variable use x.show() method          
"""

# define weights for objective terms and constraints
Q = torch.eye(17)                          # loss function weight
Q[13, 13], Q[14,14], Q[15,15], Q[16,16] = 0,0,0,0
Q_con = 100.0                    # constraints penalty weight

# variables
# x = variable("x")[:, [0]]
# y = variable("x")[:, [1]]
x = variable('state')
u = variable('input')
s_dot = variable([x, u], quad.state_dot, 'state_dot')

# evaluate the graph
bs = 100
out = s_dot({'state': torch.randn(bs,17), 'input': torch.randn(bs,4)})

env = Sim(
    ### parameters for both backends
    dt=dt,
    Ti=Ti,
    Tf=Tf,
    params=quad.params,
    backend=backend,
    init_state=state,
    reference=reference,
    integrator_type=integrator_type,
    ### eom specific arguments
    state_dot=quad.state_dot,
)

# x, y, z, q0, q1, q2, q3, xdot, ydot, zdot, p, q, r, wM1, wM2, wM3, wM4 = torch.unbind(state, dim=1)

env.set_state(state)

cmd = torch.zeros(4)

env.step(cmd)

env.state.show()



# sampled parameters
p = variable('p')


