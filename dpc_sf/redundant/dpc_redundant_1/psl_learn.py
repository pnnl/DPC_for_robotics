# copying from auto_system_train.py

import os

import sklearn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from neuromancer.modules.blocks import MLP
from neuromancer.dynamics.integrators import Euler
from neuromancer.psl.autonomous import systems
from neuromancer.problem import Problem
from neuromancer.loggers import MLFlowLogger
from neuromancer.trainer import Trainer
from neuromancer.constraint import Loss, variable
from neuromancer.loss import PenaltyLoss
from neuromancer.dataset import DictDataset
from neuromancer.callbacks import Callback
from neuromancer.system import Node, System

from neuromancer.psl.base import ODE_NonAutonomous
from neuromancer.psl.base import cast_backend
from neuromancer.psl.signals import step

from dpc_sf.dynamics.params import params
from dpc_sf.gym_environments import multirotor_utils

import dpc_sf.utils.pytorch_utils as ptu

def truncated_mse(true, pred):
    diffsq = (true - pred) ** 2
    truncs = diffsq > 1.0
    tmse = truncs * np.ones_like(diffsq) + ~truncs * diffsq
    return tmse.mean()

class TSCallback(Callback):
    def __init__(self, validator, logdir):
        self.validator = validator
        self.logdir = logdir

    def begin_eval(self, trainer, output):
        tmse, mse, sim, real = self.validator()
        output['eval_tmse'] = tmse
        output['eval_mse'] = mse
        # plot_traj(real, sim, figname=f'{self.logdir}/{self.validator.figname}steps_open.png')
        # plt.close()

class Validator:

    def __init__(self, netG, sys, args, normalize=False):
        """
        Used for evaluating model performance

        :param netG: (nn.Module) Some kind of neural network state space model
        :param sys: (psl.ODE_NonAutonomous) Ground truth ODE system
        :param normalize: (bool) Whether to normalized data. Will denorm before plotting and loss calculation if True
        """
        self.figname = ''
        self.x0s = [sys.get_x0() for i in range(10)]
        self.normalize = normalize
        self.sys = sys
        X = []
        for x0 in self.x0s:
            sim = sys.simulate(ts=args.ts, nsim=1000, x0=x0)
            X.append(torch.tensor(sim['X'], dtype=torch.float32))
            if normalize:
                X[-1] = sys.normalize(X[-1], key='X')

        def mse_loss(input, target):
            return torch.mean((input - target) ** 2, axis=(1, 2))

        self.mse = mse_loss

        self.reals = {'X': torch.stack(X)}
        self.netG = netG

    def __call__(self):
        """
        Runs the model in it's current state on 10 validation set trajectories with
        different initial conditions and control sequences.

        :return: truncated_mse, mse over all validation rollouts
        and x, x_pred trajectories for best validation rollout
        """
        with torch.no_grad():
            self.reals['xn'] = self.reals['X'][:, 0:1, :]
            simulation = self.netG.forward(self.reals)
            x = self.reals['X']
            xprime = simulation['xn'][:, :-1, :]
            xprime = torch.nan_to_num(xprime, nan=200000., posinf=None, neginf=None)
            if self.normalize:
                x = self.sys.denormalize(x, key='X')
                xprime = self.sys.denormalize(xprime, key='X')
            print('dev', x.shape, xprime.shape)
            mses = self.mse(x, xprime)
            truncs = truncated_mse(x, xprime)
        best = np.argmax(mses)
        return truncs.mean(), mses.mean(), xprime[best].detach().numpy(), x[best].detach().numpy()

def get_data(nsteps, sys, nsim, bs, normalize=False):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.ODE_NonAutonomous)
    :param normalize: (bool) Whether to normalize the data

    """
    X = []
    for _ in range(nsim):
        sim = sys.simulate(nsim=nsteps, x0=sys.get_x0())
        X.append(sim['X'])

    X = np.stack(X)
    sim = sys.simulate(nsim=nsim * nsteps, x0=sys.get_x0())

    nx = X.shape[-1]
    x = sim['X'].reshape(nsim, nsteps, nx)
    X = np.concatenate([X, x], axis=0)

    X = torch.tensor(X, dtype=torch.float32)
    if normalize:
            X = sys.normalize(X, key='X')
    train_data = DictDataset({'X': X, 'xn': X[:, 0:1, :]}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                              collate_fn=train_data.collate_fn, shuffle=True)
    dev_data = DictDataset({'X': X[0:1], 'xn': X[0:1, 0:1, :]}, name='dev')
    dev_loader = DataLoader(dev_data, num_workers=1, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=False)
    test_loader = dev_loader
    return nx, train_loader, dev_loader, test_loader

class EulerIntegrator(nn.Module):
    """
    Simple black-box NODE
    """
    def __init__(self, nx, hsize, nlayers, ts):
        super().__init__()
        self.dx = MLP(nx, nx, bias=True, linear_map=nn.Linear, nonlin=nn.ELU,
                      hsizes=[hsize for h in range(nlayers)])
        self.integrator = Euler(self.dx, h=torch.tensor(ts))

    def forward(self, xn):
        """

        :param xn: (Tensor, shape=(batchsize, nx)) State
        :param u: (Tensor, shape=(batchsize, nu)) Control action
        :return: (Tensor, shape=(batchsize, nx)) xn+1
        """
        return self.integrator(xn)


def get_node(nx, args):
    integrator = EulerIntegrator(nx, args.hsize, args.nlayers, torch.tensor(args.ts))
    nodes = [Node(integrator, ['xn'], ['xn'])]
    system = System(nodes)
    return system


class QuadcopterPSL(ODE_NonAutonomous):
    """
    Two Tank model.
    `Original code obtained from APMonitor <https://apmonitor.com/do/index.php/Main/LevelControl>`_
    """
    @property
    def params(self, params=params):
        variables = {
            'x0': params["default_init_state_list"],
            'block': MLP(21, 4, hsizes=[64]*3), # unknown delta from true dynamics
        }
        constants = {'ts': 0.1}
        parameters = {} 
        meta = {}
        return variables, constants, parameters, meta
    
    @cast_backend
    def get_x0(
            self,
            disorient=True,
            sample_SO3=False,
            init_max_vel=0.5,
            env_bounding_box=2.0,
            init_max_attitude=np.pi/3.0,
            init_max_angular_vel=0.1*np.pi
        ):
        """
        Method to initial the robot in Simulation environment.

        Args:
            randomize (bool): If ``True``, initialize the robot randomly.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Tuple containing the following vectors in given order:
                - qpose_init (numpy.ndarray): Vector of robot's state after perturbation (dim-18).
                - qvel_init (numpy.ndarray): Vector of robot's velocity after perturbation (dim-6).
        """

        # attitude (roll pitch yaw)
        quat_init = np.array([1., 0., 0., 0.])
        if disorient and sample_SO3:
            rot_mat = multirotor_utils.sampleSO3()
            quat_init = multirotor_utils.rot2quat(rot_mat)
        elif disorient:
            attitude_euler_rand = self.rng.uniform(low=-init_max_attitude, high=init_max_attitude, size=(3,))
            quat_init = multirotor_utils.euler2quat(attitude_euler_rand)

        # position (x, y, z)
        c = 0.2
        ep = self.rng.uniform(low=-(env_bounding_box-c), high=(env_bounding_box-c), size=(3,))
        pos_init = ep


        # velocity (vx, vy, vz)
        vel_init = multirotor_utils.sample_unit3d() * init_max_vel

        # angular velocity (wx, wy, wz)
        angular_vel_init = multirotor_utils.sample_unit3d() * init_max_angular_vel

        # omegas 
        omegas_init = np.array([self.params['w_hover']]*4)

        state_init = ptu.from_numpy(np.concatenate([pos_init, quat_init, vel_init, angular_vel_init, omegas_init]).ravel())

        return state_init

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        u = step(nsim=nsim, d=2, min=0., max=0.4, randsteps=int(np.ceil(self.ts*nsim/30)), rng=self.rng)
        return u

    @cast_backend
    def equations(self, t, x, u):

        h1 = self.B.core.clip(x[0], 0, 1)  # States (2): level in the tanks
        h2 = self.B.core.clip(x[1], 0, 1)
        pump = self.B.core.clip(u[0], 0, 1)  # Inputs (2): pump and valve
        valve = self.B.core.clip(u[1], 0, 1)
        dhdt1 = self.c1 * (1.0 - valve) * pump - self.c2 * self.B.core.sqrt(h1)
        dhdt2 = self.c1 * valve * pump + self.c2 * self.B.core.sqrt(h1) - self.c2 * self.B.core.sqrt(h2)
        if h1 >= 1.0 and dhdt1 > 0.0:
            dhdt1 = 0
        if h2 >= 1.0 and dhdt2 > 0.0:
            dhdt2 = 0
        dhdt = [dhdt1, dhdt2]
        return dhdt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-system', default='LorenzSystem', choices=[k for k in systems],
                        help='You can use any of the systems from psl.nonautonomous with this script')
    parser.add_argument('-epochs', type=int, default=100,
                        help='Number of epochs of training.')
    parser.add_argument('-normalize', action='store_true', help='Whether to normalize data')
    parser.add_argument('-lr', type=float, default=0.001,
                        help='Learning rate for gradient descent.')
    parser.add_argument('-nsteps', type=int, default=4,
                        help='Prediction horizon for optimization objective. During training will roll out for nsteps from and initial condition')
    parser.add_argument('-batch_size', type=int, default=100)
    parser.add_argument('-nsim', type=int, default=1000,
                        help="The script will generate an nsim long time series for training and testing and 10 nsim long time series for validation")
    parser.add_argument('-logdir', default='test',
                        help='Plots and best models will be saved here. Also will be moved to the location directory for mlflow artifact logging')
    parser.add_argument("-exp", type=str, default="test",
                        help="Will group all run under this experiment name.")
    parser.add_argument("-location", type=str, default="mlruns",
                        help="Where to write mlflow experiment tracking stuff")
    parser.add_argument("-run", type=str, default="neuromancer",
                        help="Some name to tell what the experiment run was about.")
    parser.add_argument('-hsize', type=int, default=128, help='Size of hiddens states')
    parser.add_argument('-nlayers', type=int, default=4, help='Number of hidden layers for MLP')
    parser.add_argument('-iterations', type=int, default=3,
                        help='How many episodes of curriculum learning by doubling the prediction horizon and halving the learn rate each episode')
    parser.add_argument('-eval_metric', type=str, default='eval_mse')
    args = parser.parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    # we need to wrap the state_dot system in the psl.nonautonomous.ODE subclass

    sys = QuadcopterPSL()
    args.ts = sys.ts
    nx, train_data, dev_data, test_data = get_data(args.nsteps, sys, args.nsim, args.batch_size, normalize=args.normalize)
