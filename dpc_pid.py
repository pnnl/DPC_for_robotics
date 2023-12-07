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
from utils.rotation import quaternion_derivative, quaternion_error, euler_to_quaternion
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
        # self.x_range = 3.0
        # self.r_range = 3.0

    def get_dataset(self):
        # we will generate random orientation quaternion for start and end positions

        # Random Euler angles for start and end points
        init_euler = 2 * torch.pi * torch.rand([self.batch_size, 3])
        init_quaternion = euler_to_quaternion.pytorch_batched(init_euler)
        init_omega = 2 * (torch.rand([self.batch_size, 3]) - 0.5)
        X = torch.hstack([init_quaternion, init_omega]).unsqueeze(1)

        reference_euler = 2 * torch.pi * torch.rand([self.batch_size, 3])
        reference_quaternion = euler_to_quaternion.pytorch_batched(reference_euler)
        reference_omega = torch.zeros([self.batch_size, 3])
        R = torch.hstack([reference_quaternion, reference_omega])
        R = torch.stack([R]*self.nstep, dim=1) # repeat the reference for the nsteps

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
        self.Ixx, self.Iyy, self.Izz, self.Ixy, self.Ixz, self.Iyz = 1.0, 1.0, 1.0, 0, 0, 0

        # Inertia matrix
        self.I = ptu.tensor([[self.Ixx, self.Ixy, self.Ixz],
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

        # Calculate rate of change of quaternion based on instantaneous angular velocity
        q_dot = quaternion_derivative.pytorch_batched(q=q, omega=omega)

        # Calculate the angular momentum L = I * omega
        L = (self.I @ omega.unsqueeze(-1)).squeeze(-1)

        # Derivative of the angular momentum
        dL_dt = u - torch.cross(omega, L)

        # Calculate rate of change of angular velocity w.r.t body axes
        omega_dot = (self.I_inv @ dL_dt.unsqueeze(-1)).squeeze(-1)

        return torch.hstack([q_dot, omega_dot])

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
    Q = 5.00
    R = 0.1

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    images_path = media_save_path + current_datetime + '/'

    # Check if directory exists, and if not, create it
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # NeuroMANCER System Definition
    # -----------------------------
    nx = 7 # state size
    nu = 3 # input size

    # Variables:
    r = nm.constraint.variable('R')           # the reference
    u = nm.constraint.variable('U')           # the input
    x = nm.constraint.variable('X')           # the state
    y = nm.constraint.variable('Y')           # the observation

    node_list = []

    process_policy_input = lambda x, r: torch.hstack([quaternion_error.pytorch_vectorized(x[:,:4], r[:,:4]), r[:,4:] - x[:,4:]])
    def process_policy_input(x, r):
        return torch.hstack([quaternion_error.pytorch_vectorized(x[:,:4], r[:,:4]), r[:,4:] - x[:,4:]])
    process_policy_input_node = nm.system.Node(process_policy_input, ['X', 'R'], ['Y'], name='preprocess')
    node_list.append(process_policy_input_node)

    policy = nm.modules.blocks.MLP(
        insize=nx, outsize=nu, bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ReLU,
        hsizes=[20, 20, 20, 20]
    ).to(ptu.device)
    policy_node = nm.system.Node(policy, ['Y'], ['U'], name='policy')
    node_list.append(policy_node)

    dynamics = AttitudeDynamics(insize=11, outsize=8)
    integrator = integrators.Euler(dynamics, h=ptu.tensor(Ts))
    dynamics_node = nm.system.Node(integrator, ['X', 'U'], ['X'], name='dynamics')
    node_list.append(dynamics_node)

    print(f'node list used in cl_system: {node_list}')
    cl_system = nm.system.System(node_list)

    dataset = AttitudeDatasetGenerator(
        batch_size = batch_size,
        minibatch_size = minibatch_size,
        nstep = nstep,
        Ts = Ts,
    )

    # Define Constraints: - none for this situation
    constraints = []

    # Define Loss:
    objectives = []

    action_loss = R * (u == ptu.tensor(0.)) ^ 2  # control penalty
    action_loss.name = 'action_loss'
    objectives.append(action_loss)

    quad_loss = Q * (y == r) ^ 2 # (x[:,:,:4] == r[:,:,:4])^2
    quad_loss.name = 'quad_loss'
    objectives.append(quad_loss)

    # objectives = [action_loss, pos_loss, vel_loss]
    loss = nm.loss.BarrierLoss(objectives, constraints)

    # Define the Problem and the Trainer:
    problem = nm.problem.Problem([cl_system], loss, grad_inference=True)
    optimizer = torch.optim.Adagrad(policy.parameters(), lr=lr)

    # Custom Callack Setup
    # --------------------
    # callback = utils.callback.LinTrajCallback(save_dir=current_datetime, media_path=media_save_path, nstep=nstep, nx=nx, Ts=Ts)
    
        # Perform the Training
    # --------------------
    for i in range(iterations):
        print(f'training with prediction horizon: {nstep}, lr: {lr}')

        # Get First Datasets
        # ------------
        dataset.nstep = nstep
        train_loader, dev_loader = dataset.get_loaders()

        trainer = nm.trainer.Trainer(
            problem,
            train_loader,
            dev_loader,
            dev_loader,
            optimizer,
            # callback=callback,
            epochs=epochs,
            patience=epochs,
            train_metric="train_loss",
            dev_metric="dev_loss",
            test_metric="test_loss",
            eval_metric='dev_loss',
            warmup=400,
            lr_scheduler=False,
            device=ptu.device
        )

        # Train Over Nsteps
        # -----------------
        cl_system.nsteps = nstep
        best_model = trainer.train()
        trainer.model.load_state_dict(best_model)

        # Update Parameters for the Next Iteration
        # ----------------------------------------
        lr *= lr_multiplier # 0.2

        # update the prediction horizon
        cl_system.nsteps = nstep

        optimizer.param_groups[0]['lr'] = lr

    # Save the Policy
    # ---------------
    # %%
    policy_state_dict = {}
    for key, value in best_model.items():
        if "callable." in key:
            new_key = key.split("nodes.0.nodes.1.")[-1]
            policy_state_dict[new_key] = value
    torch.save(policy_state_dict, policy_save_path + f"test.pth")

class OmegadotDatasetGenerator:

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
        # self.x_range = 3.0
        # self.r_range = 3.0

    def get_dataset(self):

        pass

class OmegadotDynamics(ode.ODESystem):
    def __init__(self, insize, outsize) -> None:
        super().__init__(insize=insize, outsize=outsize)

        self.in_features = insize
        self.out_features = outsize

        self.Ixx, self.Iyy, self.Izz, self.Ixy, self.Ixz, self.Iyz = 1.0, 1.0, 1.0, 0, 0, 0

        # Inertia matrix
        self.I = ptu.tensor([[self.Ixx, self.Ixy, self.Ixz],
                               [self.Ixy, self.Iyy, self.Iyz],
                               [self.Ixz, self.Iyz, self.Izz]])

        # Invert the inertia matrix
        self.I_inv = torch.linalg.inv(self.I)

    def ode_equations(self, x, u):
        # x: rotors angular velocity {w1, w2, w3, w4}
        # u: rotors angular acceleration {w1dot, w2dot, w3dot, w4dot}
        pass



if __name__ == "__main__":

    ptu.init_dtype()
    ptu.init_gpu(use_gpu=False)

    train_attitude_control(iterations=2, epochs=10, batch_size=5000, minibatch_size=1000, nstep=100, lr=0.05, Ts=0.1)