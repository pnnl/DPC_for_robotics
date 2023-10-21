"""
In this script we hope to learn residuals between the equations of motion
and the mujoco - it works as far as I can tell! 

More robust with:
    - smaller MLP in eom_sysid
    - zero input always in simulate function below

Even works with RK4 mix and matched with non zero input on quad_mj and NM
"""

# Neuromancer syntax example for constrained optimization
import neuromancer as nm
import torch 
from torch.utils.data import DataLoader
import torch.optim as optim
import os

from dpc_sf.dynamics.eom_sysid import QuadcopterSysID
from dpc_sf.dynamics.eom_pt import QuadcopterPT
from dpc_sf.dynamics.mj import QuadcopterMJ
from dpc_sf.dynamics.params import params
from dpc_sf.dynamics.jac_pt import QuadcopterJac
from dpc_sf.control.trajectory.trajectory import waypoint_reference
import dpc_sf.gym_environments.multirotor_utils as utils
from dpc_sf.utils.normalisation import normalise_nm, denormalize_nm
import dpc_sf.utils.pytorch_utils as ptu
from dpc_sf.control.dpc.dpc_utils import generate_data, simulate, get_node
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
        plt.close()

class Validator:

    def __init__(self, netG, quad_mj, ts=0.1, normalize=False):
        """
        Used for evaluating model performance

        :param netG: (nn.Module) Some kind of neural network state space model
        :param sys: (psl.ODE_NonAutonomous) Ground truth ODE system
        :param normalize: (bool) Whether to normalized data. Will denorm before plotting and loss calculation if True
        """
        self.figname = ''
        self.x0s = [random_state() for i in range(300)]
        self.normalize = normalize
        self.quad_mj = quad_mj
        X, U = [], []
        for x0 in self.x0s:
            # quad_mj.start_online_render()
            sim = simulate(quad_mj, nsim=30, x0=x0)
            X.append(torch.tensor(sim['X'], dtype=torch.float32))
            U.append(torch.tensor(sim['U'], dtype=torch.float32))
        
        X = torch.stack(X)
        U = torch.stack(U)

        if normalize:
            X = normalise_nm(X, means=ptu.from_numpy(params["state_mean"]), variances=ptu.from_numpy(params["state_var"]))
            U = normalise_nm(U, means=ptu.from_numpy(params["input_mean"]), variances=ptu.from_numpy(params["input_var"]))

        def mse_loss(input, target):
            return torch.mean((input - target) ** 2, axis=(1, 2))

        self.mse = mse_loss

        self.reals = {'X': X, 'U': U}
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
                x = denormalize_nm(x, means=ptu.from_numpy(params["state_mean"]), variances=ptu.from_numpy(params["state_var"]))
                xprime = denormalize_nm(xprime, means=ptu.from_numpy(params["state_mean"]), variances=ptu.from_numpy(params["state_var"]))

            mses = self.mse(x, xprime)
        best = np.argmax(mses)
        return None, mses.mean(), xprime[best].detach().numpy(), x[best].detach().numpy()



if __name__ == '__main__':

    Ts = 0.1
    lr = 0.001
    logdir = 'test'
    epochs = 10
    iterations = 3
    eval_metric = 'eval_mse'
    nsteps = 3
    normalize = True

    # Sample rollout of true system
    # -----------------------------
    nx, nu, train_data, dev_data, test_data, quad = generate_data(
        Ts=Ts,
        nsteps=nsteps,
        nsim=10000,
        bs=32,
        quad_type='mj',
        normalize=normalize
    )

    quad_sysid = QuadcopterSysID()
    ssm = get_node(state_dot=quad_sysid, ts=Ts)
    opt = optim.Adam(ssm.parameters(), lr, betas=(0.0, 0.9))
    validator = Validator(ssm, quad, ts=Ts, normalize=normalize)
    callback = TSCallback(validator, logdir)
    objectives = []

    xpred = variable('xn')[:, :-1, :]
    xtrue = variable('X')
    loss = (xpred == xtrue) ^ 2
    loss.update_name('loss')

    obj = PenaltyLoss([loss], [])
    problem = Problem([ssm], obj)

    logout = ['loss', 'fd', 'mse_test', 'mae_test', 'r2_test', 'eval_mse']
    # logger = MLFlowLogger(args, savedir=args.logdir, stdout=['train_loss', 'eval_mse'], logout=logout)
    trainer = Trainer(problem, train_data, dev_data, test_data, opt,
        callback=callback,
        epochs=epochs,
        patience=epochs*iterations,
        train_metric='train_loss',
        dev_metric='dev_loss',
        test_metric='test_loss',
        eval_metric=eval_metric
    )

    for i in range(iterations):
        print(f'training {nsteps} objective, lr={lr}')
        validator.figname = str(nsteps)
        best_model = trainer.train()
        trainer.model.load_state_dict(best_model)
        lr /= 2.0
        nsteps *= 2
        nx, nu, train_data, dev_data, test_data, quad = generate_data(
            Ts=Ts,
            nsteps=nsteps,
            nsim=10000,
            bs=32,
            quad_type='mj',
            normalize=normalize
        )
        # nx, nu, train_data, dev_data, test_data = get_data(nsteps=nsteps, quad_mj=quad_mj, nsim=10000, bs=32, normalize=normalize) 
        trainer.train_data, trainer.dev_data, trainer.test_data = train_data, dev_data, test_data
        opt.param_groups[0]['lr'] = lr

    # Test set results
    # ----------------
    x0 = random_state()
    sim = simulate(quad=quad, nsim=30, x0=x0)
    X = torch.tensor(sim['X'], dtype=torch.float32)
    X = X.view(1, *X.shape)
    U = torch.tensor(sim['U'], dtype=torch.float32)
    U = U.view(1, *U.shape)
    if normalize:
        X = normalise_nm(X, means=ptu.from_numpy(params["state_mean"]), variances=ptu.from_numpy(params["state_var"]))
        U = normalise_nm(U, means=ptu.from_numpy(params["input_mean"]), variances=ptu.from_numpy(params["input_var"]))

    true_traj = X
    with torch.no_grad():
        pred_traj = ssm.forward({'X': X, 'U': U, 'xn': X[:, 0:1, :]})['xn'][:, :-1, :]
    if normalize:
        pred_traj = denormalize_nm(pred_traj, means=ptu.from_numpy(params["state_mean"]), variances=ptu.from_numpy(params["state_var"]))
        true_traj = denormalize_nm(true_traj, means=ptu.from_numpy(params["state_mean"]), variances=ptu.from_numpy(params["state_var"]))

    pred_traj = pred_traj.detach().numpy().reshape(-1, nx)
    true_traj = true_traj.detach().numpy().reshape(-1, nx)

    np.save(os.path.join(logdir, f'test_true_loop.npy'), true_traj)
    np.save(os.path.join(logdir, f'test_pred_loop.npy'), pred_traj)

    pred_traj, true_traj = pred_traj.transpose(1, 0), true_traj.transpose(1, 0)

    figsize = 25
    plt.xticks(fontsize=figsize)
    fig, ax = plt.subplots(nx, figsize=(figsize, figsize))
    labels = [f'$y_{k}$' for k in range(len(true_traj))]
    for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
        if nx > 1:
            axe = ax[row]
        else:
            axe = ax
        axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
        axe.plot(t1, label='True', c='c')
        axe.plot(t2, label='Pred', c='m')
        axe.tick_params(labelbottom=False, labelsize=figsize)
    axe.tick_params(labelbottom=True, labelsize=figsize)
    axe.legend(fontsize=figsize)
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, 'open_loop.png'))
    # logger.log_artifacts({})


    # try for next two days:
    # ----------------------

    # sample 50 long rollouts at 0.1s
    # sinusoidal inputs to each rotor
    # normalise
    # maybe switch to RL action space

