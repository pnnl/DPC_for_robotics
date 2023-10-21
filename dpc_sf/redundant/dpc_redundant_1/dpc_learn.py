"""

"""

import neuromancer as nm
import torch 
from torch.utils.data import DataLoader
import torch.optim as optim
import os

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


import matplotlib.pyplot as plt
torch.manual_seed(0)

class Policy(torch.nn.Module):

    def __init__(self, insize, outsize, Ts=0.1):
        super().__init__()
        self.net = MLP(insize, outsize, bias=True, hsizes=[20, 20, 20])
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
        cmd_w = hover_w + action * motor_range / 2
        
        # the above is the commanded omega
        w_error = cmd_w - x[:,13:]
        p_gain = params["IRzz"] / self.Ts
        motor_inputs = w_error * p_gain
        return motor_inputs

    def forward(self, x, R):
        features = torch.cat([x, R], dim=-1)
        action = self.net(features)
        motor_inputs = self.get_motor_input(x, action)
        return motor_inputs



if __name__ == '__main__':

    Ts = 0.1
    lr = 0.0001
    logdir = 'test'
    epochs = 5000
    iterations = 3
    eval_metric = 'eval_mse'
    nsteps = 3
    normalize = True
    nx = 17
    nu = 4

    # Sample rollout of true system
    # -----------------------------
    train_loader, dev_loader, quad = generate_dpc_data(
        Ts=Ts,
        nsteps=3000,
        nsim=4, # 10_000
        bs=32,
        quad_type='eom', # 'mj', 'eom'
        normalize=True
    )

    quad = QuadcopterDPC()

    insize = 2*nx
    policy = Policy(insize, nu)
    policy_node = Node(policy, ['xn', 'R'], ['U'], name='policy')
    # system_node = get_node(state_dot=quad, ts=Ts)
    integrator = integrators.Euler(quad, h=torch.tensor(Ts), interp_u=lambda tq, t, u: u)
    system_node = Node(integrator, ['xn', 'U'], ['xn'])
    cl_system = System([policy_node, system_node])
    # cl_system.show()

    # Optimising the Control Policy
    # -----------------------------
    opt = optim.Adam(policy.parameters(), lr=lr)

    tru = variable('xn')[:, 1:, :13]
    ref = variable('R')[:, :, :13]
    u = variable('U')

    loss = (tru == params["default_init_state_pt"][:13]) ^ 2
    loss.update_name('loss')

    obj = PenaltyLoss([loss], [])
    problem = Problem([cl_system], obj)

    logout = ['loss']
    trainer = Trainer(problem, train_loader, dev_loader, dev_loader, opt,
                    epochs=epochs,
                    patience=1000,
                    train_metric='train_loss',
                    dev_metric='dev_loss',
                    test_metric='dev_loss',
                    eval_metric='dev_loss')

    best_model = trainer.train()
    
    trainer.model.load_state_dict(best_model)

    # evaluate the trained model
    # --------------------------

    quad_test = QuadcopterPT()
    with torch.no_grad():
        policy = policy_node.callable
        for i in range(50):
            t = quad_test.t
            R = ptu.from_numpy(normalize_np(quad_test.reference(t))).unsqueeze(0)
            x = ptu.from_numpy(normalize_np(quad_test.get_state())).unsqueeze(0)
            loss = (x[:13] - R[:13]) @ (x[:13] - R[:13]).T

            U = ptu.to_numpy(policy(x, R))
            quad_test.step(U)
    quad_test.animate()

    print('fin')

    # --------------------------
    # We will have to denormalize the policy actions according to the system stats
    # Conversely we will have to normalize the system states according to the system stats to hand to the policy

    def norm(x):
        if isinstance(x, np.ndarray):
            return ptu.from_numpy(normalize_np(x))
        elif isinstance(x, torch.Tensor):
            return normalise_nm(x)

    def denorm(u):
        return u # denormalize_nm(u)

    normnode = Node(norm, ['xsys'], ['xn'], name='norm')
    # policy_node = Node(policy, ['xn', 'R'], ['U'], name='policy')
    # denormnode = Node(denorm, ['U'], ['u'], name='denorm')
    # sysnode = Node(, ['xsys', 'u'], ['xsys'], name='actuator')
    integrator = integrators.Euler(quad, h=torch.tensor(Ts), interp_u=lambda tq, t, u: u)
    system_node = Node(integrator, ['xsys', 'U'], ['xsys'])
    test_system = System([normnode, policy_node, system_node])
    # test_system.show()

    from neuromancer.psl.signals import sines, step, arma, spline
    import numpy as np
    references = spline(nsim=1000, d=nx, min=params["rl_min"], max=params["rl_max"])
    plt.plot(references)
    test_data = {'R': torch.tensor(normalize_np(references), dtype=torch.float32).unsqueeze(0), 'xsys': ptu.from_numpy(random_state().reshape(1, 1, -1)),
                'Time': (np.arange(1000)*Ts).reshape(1, 1000, 1)}
    
    print({k: v.shape for k, v in test_data.items()})

    test_system.nsteps=1000
    with torch.no_grad():
        test_out = test_system(test_data)

    print({k: v.shape for k, v in test_out.items()})
    fix, ax = plt.subplots(nrows=3)
    for v in range(3):
        ax[v].plot(test_out['xn'][0, 1:, v].detach().numpy(), label='pred')
        ax[v].plot(test_data['R'][0, :, v].detach().numpy(), label='true')
    plt.legend()
    plt.savefig('control.png')


    print('fin')
