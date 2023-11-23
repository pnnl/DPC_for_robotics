"""
This file aims to replace the low level control with DPC in different parts
of the quads relative degrees
"""
import os
from datetime import datetime
import torch
import numpy as np
import neuromancer as nm
from neuromancer.dynamics import ode, integrators

import utils.pytorch as ptu
import utils.callback
import reference
from utils.quad import Animator
from utils.rotation import quaternion_derivative, quaternion_error, quaternion_multiply, euler_to_quaternion, quaternion_derivative_to_angular_velocity
from dynamics import mujoco_quad, get_quad_params
from pid import PID, get_ctrl_params

class AttitudeDatasetGenerator:
    def __init__(
            self,
            batch_size,     # 5000
            minibatch_size, # 10
            nstep,          # 100
            Ts,             # 0.001
        ) -> None:

        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.nstep = nstep
        self.Ts = Ts

        # parameters which will not change:
        self.shuffle_dataloaders = False
        self.x_range = 3.0
        self.r_range = 3.0
        self.nx = 8

    def get_dataset(self):
        # we will generate random orientation quaternion for start and end positions

        # Random Euler angles for start and end points
        initial_condition_euler_angles = 2 * torch.pi * torch.rand([self.batch_size, self.nx])
        X = euler_to_quaternion.pytorch_batched(initial_condition_euler_angles).unsqueeze(1)
        reference_euler_angles = 2 * torch.pi * torch.rand([self.batch_size, self.nx])
        R = euler_to_quaternion.pytorch_batched(reference_euler_angles).unsqueeze(1)
        R = torch.cat([R]*self.nstep, dim=1)

        return {'X': X, 'R': R}
    
    def get_dictdatasets(self):

        train_data = nm.dataset.DictDataset(self.get_dataset(), name='train')
        dev_data = nm.dataset.DictDataset(self.get_dataset(), name='dev')

        return train_data, dev_data
    
    def get_loaders(self):

        train_data, dev_data = self.get_dictdatasets()
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.minibatch_size,
                                                collate_fn=train_data.collate_fn, shuffle=self.shuffle_dataloaders)
        dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=self.minibatch_size,
                                                collate_fn=dev_data.collate_fn, shuffle=self.shuffle_dataloaders)
        
        return train_loader, dev_loader

class AttitudeDynamics(ode.ODESystem):
    def __init__(self, insize, outsize) -> None:
        super().__init__(insize=insize, outsize=outsize)
        self.in_features = insize
        self.out_features = outsize
        self.Ixx, self.Iyy, self.Izz, self.Ixy, self.Ixz, self.Iyz = None, None, None, 0, 0, 0

        # Inertia matrix
        self.I = torch.tensor([[self.Ixx, self.Ixy, self.Ixz],
                               [self.Ixy, self.Iyy, self.Iyz],
                               [self.Ixz, self.Iyz, self.Izz]])

        # Invert the inertia matrix
        self.I_inv = torch.linalg.inv(self.I)

    def ode_equations(self, x, u):
        # x: orientation and angular rotation {q0, q1, q2, q3, p, q, r}
        # u: external torques                 {tau_x, tau_y, tau_z}

        # retrieve quaternion and omega
        q     = x[:,:4]
        omega = x[:,4:]

        # Calculate the angular momentum L = I * omega
        L = (self.I @ omega.unsqueeze(-1)).squeeze(-1)

        # Derivative of the angular momentum
        dL_dt = u - torch.cross(omega, L)

        # Update omega using Euler's equations
        omega_dot = (self.I_inv @ dL_dt.unsqueeze(-1)).squeeze(-1)

        q_dot = quaternion_derivative.pytorch_batched(q=q, omega=omega)

        return 
    
def train_attitude_control(
    iterations,      # 2
    epochs,          # 15
    batch_size,      # 5000
    minibatch_size,  # 10
    nstep,           # 100
    lr,              # 0.05
    Ts,              # 0.1
    policy_save_path = 'data/',
    media_save_path = 'data/training/',
    ):
    
    # unchanging parameters
    lr_multiplier = 0.5
    Qpos = 5.00
    Qvel = 5.00

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    images_path = media_save_path + current_datetime + '/'

    # Check if directory exists, and if not, create it
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # NeuroMANCER System Definition
    # -----------------------------
    nx = 8 # state size
    nu = 3 # input size

    # Variables:
    r = nm.constraint.variable('R')           # the reference
    u = nm.constraint.variable('U')           # the input
    x = nm.constraint.variable('X')           # the state

    node_list = []

    process_policy_input = lambda x, r: quaternion_error.pytorch_vectorized(x, r)
    process_policy_input_node = nm.system.Node(process_policy_input, ['X', 'R'], ['Obs'], name='preprocess')
    node_list.append(process_policy_input_node)

    policy = nm.modules.blocks.MLP(
        insize=nx, outsize=nu, bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ReLU,
        hsizes=[20, 20, 20, 20]
    ).to(ptu.device)
    policy_node = nm.system.Node(policy, ['Obs'], ['U'], name='policy')
    node_list.append(policy_node)

    dynamics = AttitudeDynamics(insize=11, outsize=8)
    integrator = integrators.Euler(dynamics, h=ptu.tensor(Ts))
    dynamics_node = nm.system.Node(integrator, ['X', 'U'], ['X'], name='dynamics')
    node_list.append(dynamics_node)

    print(f'node list used in cl_system: {node_list}')
    cl_system = nm.system.System(node_list)

if __name__ == "__main__":
    train_attitude_control(iterations=1, epochs=5, batch_size=5000, minibatch_size=10, nstep=100, lr=0.05, Ts=0.1)