"""

"""

from typing import Union
import neuromancer as nm
import torch 
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from tqdm import tqdm

from dpc_sf.control.mpc.mpc import MPC_Point_Ref, MPC_Traj_Ref
from dpc_sf.dynamics.eom_ca import QuadcopterCA
from dpc_sf.dynamics.eom_dpc import QuadcopterDPC
from dpc_sf.dynamics.eom_pt import QuadcopterPT
from dpc_sf.dynamics.mj import QuadcopterMJ
from dpc_sf.dynamics.params import params
from dpc_sf.dynamics.jac_pt import QuadcopterJac
from dpc_sf.control.trajectory.trajectory import waypoint_reference
import dpc_sf.gym_environments.multirotor_utils as utils
from dpc_sf.utils.normalisation import normalise_nm, denormalize_nm, normalise_dict, normalize_np
import dpc_sf.utils.pytorch_utils as ptu
from dpc_sf.control.dpc.dpc_utils import generate_data, simulate, get_node, generate_dpc_data
from dpc_sf.utils.random_state import random_state
import numpy as np

from neuromancer.dynamics.ode import ODESystem
from neuromancer.modules.blocks import MLP
from neuromancer.system import Node, System
from neuromancer.dynamics import integrators, ode
from neuromancer.dynamics.ode import ode_param_systems_auto as systems
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.dynamics.integrators import Euler, RK4
from neuromancer.callbacks import Callback
from neuromancer.loggers import MLFlowLogger
from neuromancer.psl.base import ODE_NonAutonomous as ODE
from neuromancer.psl.base import cast_backend
from dpc_sf.dynamics.eom_pt import state_dot_nm
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.utils.random_state import random_state
from neuromancer.psl.signals import step
from neuromancer.modules.activations import soft_exp, SoftExponential, SmoothedReLU
import neuromancer.slim as slim

import matplotlib.pyplot as plt
torch.manual_seed(0)

class Policy(torch.nn.Module):

    def __init__(
            self, 
            insize=34, 
            outsize=4,
            hsize=64,
            n_layers=3,
            mlp_umin=-1,
            mlp_umax=1, 
            Ts=0.1,
            input_keys = ['X', 'R'],
            output_keys = ['U'],
            name = 'policy'
        ):
        super().__init__()
        # self.net = MLP(insize, outsize, bias=True, hsizes=[20, 20, 20])
        self.net = MLP_bounds_MIMO(
            insize=insize, 
            outsize=outsize, 
            hsizes=[hsize] * n_layers,
            min=[mlp_umin] * outsize, 
            max=[mlp_umax] * outsize
        )

        self.input_keys = input_keys
        self.output_keys = output_keys
        self.name = name

        self.mlp_range = mlp_umax - mlp_umin

        self.Ts = Ts

    def get_motor_input(self, x, action):
        """
        Transform policy actions to motor inputs.

        Args:
            action (numpy.ndarray): Actions from policy of shape (4,).

        Returns:
            numpy.ndarray: Vector of motor inputs of shape (4,).
        """
        # the below allows the RL to specify omegas between 122.98 and 922.98
        # the true range is 75 - 925, but this is just a quick and dirty translation
        # and keeps things symmetrical about the hover omega

        # action = torch.clip(action, min=torch.tensor(-2), max=torch.tensor(2))
        motor_range = 400.0 # 400.0 # 2.0
        hover_w = params['w_hover']
        cmd_w = hover_w + action * motor_range / self.mlp_range
        
        # the above is the commanded omega
        w_error = cmd_w - x[:,13:]
        p_gain = params["IRzz"] / self.Ts
        motor_inputs = w_error * p_gain
        return motor_inputs

    def forward(self, xr):
        x = xr['Y']
        r = xr['R']
        features = torch.cat([x, r], dim=-1)
        action = self.net(features)
        motor_inputs = self.get_motor_input(x, action).unsqueeze(0)
        return {'U': motor_inputs}

def sigmoid_scale(x, min, max):
    return (max - min) * torch.sigmoid(x) + min

def relu_clamp(x, min, max):
    x = x + torch.relu(-x + min)
    x = x - torch.relu(x - max)
    return x

class MLP_bounds_MIMO(MLP):
    """
    Multi-Layer Perceptron consistent with blocks interface
    """
    bound_methods = {'sigmoid_scale': sigmoid_scale,
                    'relu_clamp': relu_clamp}

    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=slim.Linear,
        nonlin=SoftExponential,
        hsizes=[64],
        linargs=dict(),
        min=[-1.0],
        max=[1.0],
        method='sigmoid_scale',
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param bias: (bool) Whether to use bias
        :param linear_map: (class) Linear map class from slim.linear
        :param nonlin: (callable) Elementwise nonlinearity which takes as input torch.Tensor and outputs torch.Tensor of same shape
        :param hsizes: (list of ints) List of hidden layer sizes
        :param linargs: (dict) Arguments for instantiating linear layer
        :param dropout: (float) Dropout probability
        """
        super().__init__(insize=insize, outsize=outsize, bias=bias,
                         linear_map=linear_map, nonlin=nonlin,
                         hsizes=hsizes, linargs=linargs)
        assert len(min) == outsize, f'min and max ({min}, {max}) should have the same size as the output ({outsize})'
        assert len(min) == len(max), f'min ({min}) and max ({max}) should be of the same size'

        self.min = torch.tensor(min)
        self.max = torch.tensor(max)
        self.method = self._set_method(method)

    def _set_method(self, method):
        if method in self.bound_methods.keys():
            return self.bound_methods[method]
        else:
            assert callable(method), \
                f'Method, {method} must be a key in {self.bound_methods} ' \
                f'or a differentiable callable.'
            return method

    def forward(self, x):
        """
        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        for lin, nlin in zip(self.linear, self.nonlin):
            x = nlin(lin(x))
        return self.method(x, self.min, self.max)

    
class QuadcopterODE(ODE):
    """
    PSL port of quadcopter simulation
    """
    def __init__(
            self, 
            ts=0.1, 
            nsim=1001,
            exclude_norms=['Time'], 
            backend='torch', 
            requires_grad=False, 
            seed: int = 59, 
            set_stats=True
        ):

        self.ts = ts
        self.nsim = nsim
        super().__init__(exclude_norms, backend, requires_grad, seed, set_stats)

    @property
    def params(self):
        variables = {'x0': quad_params["default_init_state_list"]}
        constants = {'ts': 0.1}
        parameters = {}
        meta = {}
        return variables, constants, parameters, meta
    
    @cast_backend
    def get_x0(self):
        return random_state()
    
    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        u = step(nsim=nsim, d=4, min=-1, max=1, randsteps=int(np.ceil(self.ts*nsim/30)), rng=self.rng)
        return u
    
    def equations(self, t, x, u):
        return state_dot_nm(state=x, cmd=u, params=quad_params)
    

if __name__ == '__main__':

    Ts = 0.1 # simulation timestep
    # nsim = 1001 # number of STEPS taken in the statistical sampling stage of the system init
    get_new_data = False
    save_dir = 'logs/data'
    plot_data = False
    test_closed_loop = False
    lr = 0.001
    epochs = 1000
    patience = 1000

    """
    Model
    -----

    The way that NeuroMANCER works is that it uses a 1000 step open loop simulation
    to generate statistics for the system for normalisation and perhaps other things.
    This is not great for us here as the Quadcopters open loop will become unbounded.
    
    If the psl quad has a ts and an nsim, then it gets inherited when we call super()
    and then if we call set_stats and apply the mpc state history then we can hopefully
    get an accurate set of stats for normalisation.
    """
    print(f"loading MPC trajectory...")
    mpc_trajectory = np.load(f'{save_dir}/xu_traj_eom_wp_p2p.npz')
    u_history, x_history = mpc_trajectory["u_history"], mpc_trajectory["x_history"]

    print(f"instantiating PSL QuadcopterODE...")
    sys = QuadcopterODE(ts=Ts, nsim=u_history.shape[0], set_stats=False, backend='torch')

    print(f"setting statistics for PSL QuadcopterODE using MPC trajectory...")
    sys.set_stats(
        x0 = x_history[0,:],
        U = u_history
    )

    if plot_data:
        print(f"plotting MPC reference trajectory")
        mpc_data = {
            'Y': x_history,
            'X': x_history,
            'U': u_history
        }
        sys.show(mpc_data)

    """
    Policy
    ------

    The policy needs to be clipped in a normalisation bound for effective learning. This
    was done using a new MLP class built during Soumyas submersible project.

    Further this MLP outputs desired rotor rotational rates, and a proportional controller
    decides on the true input to adjust the rates to be the desired ones.
    """

    policy = Policy(
        insize = sys.nx * 2, # state and reference
        outsize = sys.nu,        # input size
        hsize=64,
        n_layers=3,
        mlp_umin=-1,                # minimum output of MLP within policy
        mlp_umax=1,                 # maximum output of MLP within policy
        Ts=Ts,
        input_keys = ['R', 'Y'],
        output_keys = ['U'],
        name = 'policy'
    )

    """
    Dataset
    -------

    The DPC algorithm requires a dataset of starting and end positions. As to what
    we use to generate the end positions we have some wiggle room. We could:
        - use a system rollout: sys.simulate(nsim=1000)
        - use a set of reference positions using our reference trajectory

    I choose the latter here, as the quad can go off of the rails if we simply let
    it fall for 1000 timesteps, even with a small timestep.

    We must generate a dictionary with the keys: ['Y', 'X', 'xn']
        - Y and X are easy as they are the same, the state history
        - xn is the random initial conditions centered around every X position in the state space
    """
    def get_data(num_points=100000, reference=waypoint_reference('wp_traj', average_vel=1.0)):
        reference_end_time = 19.23 # really 19.23 for average_vel at 1.0
        R = []
        Y = []
        print(f"collecting {len(np.linspace(start=0.0, stop=reference_end_time, num=num_points))} samples from the reference trajectory...")
        for t in tqdm(np.linspace(start=0.0, stop=reference_end_time, num=num_points)):
            # get the reference state at the given time
            r = reference(t)
            R.append(r)
            # get the random initial state centered around the reference at that time
            Y.append(random_state() + r)

        R = np.stack(R).astype(np.float32)
        R = R.reshape(R.shape[0], 1, R.shape[1])
        
        Y = np.stack(Y).astype(np.float32)
        Y = Y.reshape(Y.shape[0], 1, Y.shape[1])

        return {'R': R, 'Y': Y}

    # the reference about which we generate data
    R = waypoint_reference('wp_traj', average_vel=1.0)

    if get_new_data:
        print(f"collecting training and dev samples...")
        train_data, dev_data = [get_data(num_points=100000, reference=R) for i in range(2)]
        np.savez(f'{save_dir}/train_data.npz', **train_data)
        np.savez(f'{save_dir}/dev_data.npz', **dev_data)
    else:
        print("loading training and dev samples...")
        train_data = {key: value for key, value in np.load(f'{save_dir}/train_data.npz').items()}
        dev_data = {key: value for key, value in np.load(f'{save_dir}/dev_data.npz').items()}
    
    # Put data dictionaries into NM DictDatasets
    train_data = DictDataset(train_data, name='train')
    dev_data = DictDataset(dev_data, name='dev')

    # Put NM Dictionaries into Torch DataLoaders
    # Shuffle false as we have already randomised start positions and want them lining up with their end positions
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_data.datadict['R'].shape[0], collate_fn=train_data.collate_fn, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=dev_data.datadict['R'].shape[0], collate_fn=dev_data.collate_fn, shuffle=False)

    print("dataset collection/loading complete.")

    """
    Define NeuroMANCER System
    -------------------------
    """

    # x = ptu.from_numpy(x_history[0,:]).unsqueeze(0)
    # u = ptu.from_numpy(u_history[0,:]).unsqueeze(0)

    integrator = lambda x, u, Ts=Ts: x + sys(x, u) * Ts
    # integrator = integrators.Euler(sys, h=torch.tensor(Ts), interp_u=lambda tq, t, u: u)
    system_node = Node(integrator, ['Y', 'U'], ['Y'])
    # quad = Node(xnext, ['X', 'U'], 'X')
    cl_system = System([policy, system_node])#, init_func=lambda x: x)

    if test_closed_loop:
        print(f"testing closed loop control...")
        # test call of closed loop system
        nstep = 3
        data = {
            'Y': ptu.from_numpy(x_history[0,:][None,:][None,:].astype(np.float32)),
            'R': torch.concatenate([ptu.from_numpy(R(5)[None,:][None,:].astype(np.float32))]*nstep, axis=1)
        }
        cl_system.nsteps = nstep
        cl_system(data)

    """
    Define Optimisation Problem
    ---------------------------

    Now we must define the NM variables which we want to optimise and setup the 
    cost function, learning rate, trainer object etc...
    """
    u = variable('U')
    x = variable('X')
    r = variable('R')

    action_loss = 0.001 * (u == 0.0) ^ 2
    tracking_loss = 5.0 * (x == r) ^ 2

    loss = PenaltyLoss(
        objectives=[action_loss, tracking_loss], 
        constraints=[]
    )

    problem = Problem(
        nodes = [cl_system],
        loss = loss
    )
    
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        dev_loader,
        optimizer,
        epochs=epochs,
        patience=patience,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric='dev_loss',
        warmup=400,
    )

    # Train model with prediction horizon of 2
    cl_system.nsteps = 2
    best_model = trainer.train()


    print('fin')