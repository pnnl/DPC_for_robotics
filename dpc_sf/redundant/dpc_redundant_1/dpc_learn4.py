"""
Changes from dpc_learn3.py:

    - The policy will change from a normalised desired motor w from the MLP
    to a desired orientation and upward velocity
    - we will use proportional controllers to accomplish this

"""

from typing import Union
import neuromancer as nm
import torch 
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from tqdm import tqdm
import copy

from dpc_sf.control.mpc.mpc import MPC_Point_Ref, MPC_Traj_Ref
from dpc_sf.dynamics.eom_ca import QuadcopterCA
from dpc_sf.dynamics.eom_dpc import QuadcopterDPC
from dpc_sf.dynamics.eom_pt import QuadcopterPT
from dpc_sf.dynamics.mj import QuadcopterMJ
from dpc_sf.dynamics.params import params
from dpc_sf.dynamics.jac_pt import QuadcopterJac
from dpc_sf.control.trajectory.trajectory import waypoint_reference
import dpc_sf.gym_environments.multirotor_utils as utils
import dpc_sf.utils.rotationConversion as rotationConversion
from dpc_sf.utils.normalisation import normalize_nm, denormalize_nm, normalise_dict, normalize_np, denormalize_np, denormalize_pt, normalize_pt
import dpc_sf.utils.pytorch_utils as ptu
from dpc_sf.control.dpc.dpc_utils import generate_data, simulate, get_node, generate_dpc_data
from dpc_sf.utils.random_state import random_state
import numpy as np
from dpc_sf.utils.animation import Animator


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
            outsize=3,
            hsize=64,
            n_layers=3,
            mlp_umin=-1,
            mlp_umax=1, 
            Ts=0.1,
            normalize=True,
            mean = None,
            var = None,
        ):

        super().__init__()

        # the MLP for learning DPC
        self.net = MLP_bounds_MIMO(
            insize=insize, 
            outsize=outsize, 
            hsizes=[hsize] * n_layers,
            min=[mlp_umin] * outsize, 
            max=[mlp_umax] * outsize,
        )

        # the low level controller for orientation
        self.tpr_norm_2_u = CtrlThrustPitchRoll(
            mlp_umax=mlp_umax,
            mlp_umin=mlp_umin,
            Ts=Ts,
            normalize=normalize,
            mean=mean,
            var=var
        )

    def forward(self, x_norm, r_norm):
        """
        Expects normalised state and reference signals
        """

        # the features being input to the policy are the same as before
        features = torch.cat([x_norm, r_norm], dim=-1)

        # the action will be a desired [thrust, pitch, roll]
        tpr_norm = self.net(features)

        # input desired action into the low level P controller to get desired w
        u = self.tpr_norm_2_u(x_norm, tpr_norm)

        return u
    
class CtrlThrustPitchRoll(torch.nn.Module):
    def __init__(   
            self,
            mlp_umax=1,
            mlp_umin=-1,         
            Ts=0.1,
            normalize=True,
            mean = None,
            var = None,
        ) -> None:

        super().__init__()

        self.normalize = normalize
        self.mlp_range = mlp_umax - mlp_umin
        
        self.Ts = Ts

        self.mean = mean
        self.var = var

        self.thr_range = quad_params["maxThr"] - quad_params["minThr"]
        self.rp_range = 2 * torch.pi
        self.motor_range = 400.0

        self.rp_mean = ptu.from_numpy(np.array([0,0]))
        self.thr_mean = ptu.from_numpy(np.array((quad_params["kTh"] * quad_params["w_hover"] ** 2)))

        self.w_p_gain = quad_params["IRzz"] / Ts
        self.r_p_gain = 1.0
        self.p_p_gain = 1.0
        
        self.C_M_r = ptu.from_numpy(np.array([1,-1,-1,1]))
        self.C_M_p = ptu.from_numpy(np.array([-1,-1,1,1]))
    
    def tpr_norm_2_u(self, x_norm, tpr_norm):
        """
        Expects normalised x, and mlp output action_norm
        """
        # desired thrust
        thr_des_av_norm = tpr_norm[:, 0:1]

        # desired roll, pitch
        rp_des_norm = tpr_norm[:, 1:3]

        # denormalise actions
        rp_des = self.rp_mean + rp_des_norm * self.rp_range / self.mlp_range
        thr_des_av = self.thr_mean + thr_des_av_norm * self.thr_range / self.mlp_range

        # # denormalise state, reference
        x = denormalize_pt(x_norm, means=ptu.from_numpy(self.mean), variances=ptu.from_numpy(self.var))

        # Calculate w For Orientation
        # ---------------------------

        # find unnormalised quat and convert to roll, pitch (rp)
        quat = x[:,3:7]

        # assert that the quaternion is NOT normalised
        assert (quat - x_norm[:,3:7]).abs().max() < 1e-8

        ypr = rotationConversion.quatToYPR_ZYX_nm(quat)

        quat_norm = quat / torch.sqrt(torch.sum(quat**2, dim=1, keepdim=True))

        # assert (rotationConversion.YPRToQuat_nm(ypr[:,0], ypr[:,1], ypr[:,2]) * 100 - - quat_norm <= 1e-5).all()
        p, r = ypr[:,1], ypr[:,2]

        # error in roll, pitch
        r_err = rp_des[:,0] - r
        p_err = rp_des[:,1] - p

        # calculate desired moments (delta thrusts)
        M_r_des = self.r_p_gain * r_err
        M_p_des = self.p_p_gain * p_err

        # calculate the differences in rotor thrusts for these moments
        thr_r_des = self.C_M_r * torch.vstack([M_r_des]*4).T
        thr_p_des = self.C_M_p * torch.vstack([M_p_des]*4).T

        # average the thrust commands for pitch and roll
        thr_rp_des = (thr_r_des + thr_p_des) / 2.0

        # convert these thrusts into w
        thr_des = torch.clip(thr_rp_des + thr_des_av, min=quad_params["minThr"] + 1e-8, max=quad_params["maxThr"] - 1e-8)
        w_des = torch.sqrt(thr_des / (quad_params["kTh"])) 

        # retrieve w
        w = x[:,13:17]

        # get error
        w_err = w_des - w

        # use perfect gain to get perfect w with u
        u = w_err * self.w_p_gain

        return u
    
    def forward(self, x_norm, tpr_norm):

        # input desired action into the low level P controller to get desired w
        u = self.tpr_norm_2_u(x_norm, tpr_norm)

        return u


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
        min=[0],
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

    
class QuadcopterODE(torch.nn.Module):
    """
    nn.Module version of dynamics
    """
    def __init__(
            self, 
            params=quad_params,
            ts = 0.1,
            mean = None,
            var = None,
            normalize = True,
        ) -> None:
        super().__init__()
        self.params = params
        self.ts = ts
        self.t = 0.0
        self.nx = len(params["default_init_state_list"])
        self.nu = len(params["input_mean"])
        self.normalize = normalize
        self.mean = mean
        self.var = var

    def forward(self, x, u):

        if self.normalize:
            x = denormalize_pt(x, means=ptu.from_numpy(self.mean), variances=ptu.from_numpy(self.var))

        xdot = state_dot_nm(state=x, cmd=u, params=self.params)
        xnext = x + xdot * self.ts

        if self.normalize:
            xnext = normalize_pt(xnext)

        if xnext.isnan().any():
            print('xnext has NaNs within it')

        return xnext
    
class Vis():
    def __init__(self) -> None:
        self.x = []
        self.u = []
        self.r = []
        self.t = []

    def save(self, x, u, r, t):
        self.x.append(x)
        self.u.append(u)
        self.r.append(r)
        self.t.append(t)

    def animate(self, x_p=None, drawCylinder=False):
        animator = Animator(
            states=np.vstack(self.x), 
            times=np.array(self.t), 
            reference_history=np.vstack(self.r), 
            reference=self.reference, 
            reference_type=self.reference.type, 
            drawCylinder=drawCylinder,
            state_prediction=x_p
        )
        animator.animate() # does not contain plt.show()        

if __name__ == '__main__':

    Ts = 0.1 # simulation timestep
    # nsim = 1001 # number of STEPS taken in the statistical sampling stage of the system init
    save_dir = 'logs/data'
    test_closed_loop = False
    test_sampled_data = False
    test_trained_closed_loop = True
    normalize = True
    lr = 0.01 # 0.001
    epochs = 50
    patience = 20
    train_nsteps = 5 # number of steps to take in each training rollout
    nx = 17
    nu = 4
    num_train_samples = 10_000
    num_dev_samples = 10_000
    iterations = 2
    hsize = 64
    n_layers = 3

    """
    Dataset
    -------

    The DPC algorithm requires a dataset of starting points and references. 

    We must generate a dictionary with the keys: ['X', 'R']
        - X is the random initial conditions, which I choose to be about a trajectory reference I have created
            shape: (batch size, 1, state size)
        - R is a time varying reference
            shape: (batch size, train rollout length (train_nsteps), state size)
    """
    def get_data(num_points=100000, nsteps=3, reference=waypoint_reference('wp_traj', average_vel=1.0)):
        reference_end_time = 19.23 # really 19.23 for average_vel at 1.0
        R = []
        X = []
        print(f"collecting {len(np.linspace(start=0.0, stop=reference_end_time, num=num_points))} samples from the reference trajectory...")
        for t in tqdm(np.linspace(start=0.0, stop=reference_end_time, num=num_points)):
            # get the reference state at the given time
            r = reference(t)
            R.append(r)
            # get the random initial state centered around the reference at that time
            # we set the initial quaternion to always be pointed forwards for now
            x = random_state() + r
            x[3:7] = random_state()[3:7]
            X.append(x)

        R = np.stack(R).astype(np.float32)
        R = R.reshape(R.shape[0], 1, R.shape[1])

        # we repeat R nsteps times for waypoint tracking training
        R = np.concatenate([R]*nsteps, axis=1)
        
        X = np.stack(X).astype(np.float32)
        X = X.reshape(X.shape[0], 1, X.shape[1])

        return {'R': R, 'X': X}

    # the reference about which we generate data
    R = waypoint_reference('wp_traj', average_vel=1.0)



    # Normalisation
    # -------------

    def get_stats(train_data, w_mean, w_var):
        # gather statistics on the training data
        train_data_stats = {}
        for key, value in train_data.items():
            train_data_stats[key] = {
                'mean': [],
                'var': []
            }
            for state_idx in range(nx):
                mean = value[:,:,state_idx].mean()
                var = value[:,:,state_idx].var()
                train_data_stats[key]['mean'].append(mean)
                train_data_stats[key]['var'].append(var)
            train_data_stats[key]['mean'] = np.stack(train_data_stats[key]['mean'])
            train_data_stats[key]['var'] = np.stack(train_data_stats[key]['var'])

        # want to normalise both on the starting point dataset
        mean = train_data_stats['X']['mean']
        var = train_data_stats['X']['var']

        mean[13:] = w_mean
        var[13:] = w_var

        # further we don't want to normalise the quaternion for fear of breaching limits
        mean[3:7] = np.array([0,0,0,0])
        var[3:7] = np.array([1,1,1,1])

        return mean, var

    def get_dictdataset(train_data, dev_data, mean=None, var=None, normalize=True):
        # normalise the train, dev data
        if normalize:
            train_data = {key: normalize_nm(value, means=mean, variances=var) for key, value in train_data.items()}
            dev_data = {key: normalize_nm(value, means=mean, variances=var) for key, value in dev_data.items()}

        # put datasets into torch
        train_data = {key: ptu.from_numpy(value) for key, value in train_data.items()}
        dev_data = {key: ptu.from_numpy(value) for key, value in dev_data.items()}

        # Put data dictionaries into NM DictDatasets
        train_data = DictDataset(train_data, name='train')
        dev_data = DictDataset(dev_data, name='dev')

        return train_data, dev_data
    
    def get_loader(train_data, dev_data):

        # Put NM Dictionaries into Torch DataLoaders
        # Shuffle false as we have already randomised start positions and want them lining up with their end positions
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_data.datadict['R'].shape[0], collate_fn=train_data.collate_fn, shuffle=False)
        dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=dev_data.datadict['R'].shape[0], collate_fn=dev_data.collate_fn, shuffle=False)

        return train_loader, dev_loader

    train_data = get_data(num_points=num_train_samples, nsteps=train_nsteps, reference=R)
    dev_data = get_data(num_points=num_dev_samples, nsteps=train_nsteps, reference=R)

    w_mean = quad_params["w_mean"]
    w_var = quad_params["w_var"]
    mean, var = get_stats(train_data=train_data, w_mean=w_mean, w_var=w_var)

    train_data, dev_data = get_dictdataset(train_data, dev_data, mean=mean, var=var, normalize=True)
    train_loader, dev_loader = get_loader(train_data, dev_data)

    print("done")

    """
    Model
    -----

    Now that I am not using the PSL system, normalisation will have to be done manually.
    This is achieved through the RL normalisation statistics that I have previously saved.

    These are contained in the following functions:
        - normalise_nm
        - normalise_dict
        - denormalize_nm
    """
    sys = QuadcopterODE(
        params=quad_params,
        ts=Ts,
        normalize=normalize,
        mean=mean,
        var=var
    )
    sys_node = Node(sys, input_keys=['X', 'U'], output_keys=['X'])

    """
    Policy
    ------

    The policy needs to be clipped in a normalisation bound for effective learning. This
    was done using a new MLP class built during Soumyas submersible project.

    Further this MLP outputs desired rotor rotational rates, and a proportional controller
    decides on the true input to adjust the rates to be the desired ones.
    """
    if not normalize:
        w_mean=None
        w_var=None

    policy = Policy(
        insize = sys.nx * 2, # state and reference
        outsize = sys.nu,        # input size
        hsize=hsize,
        n_layers=n_layers,
        mlp_umin=-1,                # minimum output of MLP within policy
        mlp_umax=0.90,                 # maximum output of MLP within policy
        Ts=Ts,
        normalize=normalize,        # whether or not policy expects a normalized state
        mean=mean,
        var=var
    )
    policy_node = Node(policy, input_keys=['X', 'R'], output_keys=['U'])

    """
    Define NeuroMANCER System
    -------------------------
    """
    cl_system = System([policy_node, sys_node], nsteps=train_nsteps)#, init_func=lambda x: x)

    if test_closed_loop:
        print(f"testing closed loop control...")
        # test call of closed loop system
        nstep = 3
        if normalize:
            data = {
                'X': ptu.from_numpy(normalize_nm(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32), means=mean, variances=var)),
                'R': torch.concatenate([ptu.from_numpy(normalize_nm(R(5)[None,:][None,:].astype(np.float32), means=mean, variances=var))]*nstep, axis=1)
            }
        else:
            data = {
                'X': ptu.from_numpy(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32)),
                'R': torch.concatenate([ptu.from_numpy(R(5)[None,:][None,:].astype(np.float32))]*nstep, axis=1)
            }
        cl_system.nsteps = nstep
        cl_system(data)
        print("done")

    if test_sampled_data:
        print(f"testing sampled data on closed loop system in inference...")
        cl_system.nsteps = train_nsteps
        test_data = copy.deepcopy(train_data.datadict)
        cl_system(test_data)
        print("done")

    """
    Define Optimisation Problem
    ---------------------------

    Now we must define the NM variables which we want to optimise and setup the 
    cost function, learning rate, trainer object etc...
    """
    u = variable('U')
    x = variable('X')
    r = variable('R')

    # action_loss = 0.001 * (u == 0.0) ^ 2

    # we only care about matching references after the first one
    # we also only care about first 13 states (not rotor rotational rates)
    tracking_loss = 5.0 * (x[:,1:,10:13] == r[:,:,10:13]) ^ 2
    # tracking_loss = 5.0 * (x[:,1:,7:10] == r[:,:,7:10]) ^ 2

    loss = PenaltyLoss(
        objectives=[tracking_loss], 
        constraints=[]
    )

    problem = Problem(
        nodes = [cl_system],
        loss = loss
    )
    
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    # Train model with prediction horizon of train_nsteps
    for i in range(iterations):
        print(f'training with prediction horizon: {train_nsteps}, lr: {lr}')

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

        best_model = trainer.train()
        trainer.model.load_state_dict(best_model)
        lr /= 2.0
        # train_nsteps *= 2

        # update the prediction horizon
        cl_system.nsteps = train_nsteps

        # get another set of data
        train_data = get_data(num_points=num_train_samples, nsteps=train_nsteps, reference=R)
        dev_data = get_data(num_points=num_dev_samples, nsteps=train_nsteps, reference=R)
        train_data, dev_data = get_dictdataset(train_data, dev_data, mean=mean, var=var, normalize=True)
        train_loader, dev_loader = get_loader(train_data, dev_data)

        # apply new training data and learning rate to trainer
        trainer.train_data, trainer.dev_data, trainer.test_data = train_data, dev_data, dev_data
        optimizer.param_groups[0]['lr'] = lr

    if test_trained_closed_loop:

        # Test best model with prediction horizon of 50
        problem.load_state_dict(best_model)
        nstep = 50
        if normalize:
            data = {
                'X': ptu.from_numpy(normalize_nm(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32), means=mean, variances=var)),
                'R': torch.concatenate([ptu.from_numpy(normalize_nm(R(5)[None,:][None,:].astype(np.float32), means=mean, variances=var))]*nstep, axis=1)
            }
        else:
            data = {
                'X': ptu.from_numpy(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32)),
                'R': torch.concatenate([ptu.from_numpy(R(1)[None,:][None,:].astype(np.float32))]*nstep, axis=1)
            }

        cl_system.nsteps = nstep
        trajectories = cl_system(data)
        trajectories = {key: denormalize_nm(ptu.to_numpy(value)) for key, value in trajectories.items()}

        x_p = None
        drawCylinder = False

        t = np.linspace(0, nstep*Ts, nstep)

        animator = Animator(
            states=denormalize_np(ptu.to_numpy(trajectories['X'].squeeze())), 
            times=t, 
            reference_history=denormalize_np(ptu.to_numpy(trajectories['R'].squeeze())), 
            reference=R, 
            reference_type='wp_p2p', 
            drawCylinder=drawCylinder,
            state_prediction=x_p
        )
        animator.animate() # does not contain plt.show()    

    print('fin')