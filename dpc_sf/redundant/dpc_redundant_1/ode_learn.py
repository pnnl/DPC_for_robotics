"""
In this script we hope to learn the dynamics of the quad and then test
it relative to the true dynamics using the nm.ode.py script.
"""
# Neuromancer syntax example for constrained optimization
import neuromancer as nm
import torch 
from torch.utils.data import DataLoader

from dpc_sf.redundant.eom_nm import QuadcopterNM
from dpc_sf.dynamics.eom_pt import QuadcopterPT
from dpc_sf.dynamics.mj import QuadcopterMJ
from dpc_sf.dynamics.params import params
from dpc_sf.dynamics.jac_pt import QuadcopterJac
from dpc_sf.control.trajectory.trajectory import waypoint_reference
import dpc_sf.gym_environments.multirotor_utils as utils

import dpc_sf.utils.pytorch_utils as ptu
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

import matplotlib.pyplot as plt


torch.manual_seed(0)

# ODESystem wrapper for quad.state_dot
class QuadcopterParam(ODESystem):
    def __init__(self, state_dot, insize=17+4, outsize=4):
        super().__init__(insize, outsize)
        self.state_dot = state_dot

        # recieves the velocity and tries to find residual
        self.pos_block = MLP(
            insize=3,
            outsize=3,
            hsizes=[32,32,32]
        )

        # recieves the ang vel and tries to find residual of them
        self.quat_block = MLP(
            insize=3,
            outsize=4,
            hsizes=[32,32,32]
        )
        
        # recieves the rotor omegas and tries correct body rates
        self.av_block = MLP(
            insize=4,
            outsize=3,
            hsizes=[32, 32, 32]
        )

    def ode_equations(self, x):

        # prior knowledge
        state_dot_hat = self.state_dot(x)

        # correct the updates
        pos_correction = self.pos_block(x[:, 7:10])
        quat_correction = self.quat_block(x[:, 10:13])
        av_correction = self.av_block(x[:,13:17])

        state_dot = state_dot_hat

        def ensure_2d(tensor):
            if len(tensor.shape) == 1:
                tensor = torch.unsqueeze(tensor, 0)
            return tensor
        
        state_dot = ensure_2d(state_dot)

        state_dot[:,0:3] += pos_correction
        state_dot[:,3:7] += quat_correction
        state_dot[:,10:13] += av_correction

        # zeros for inputs which do not change
        input_dot = torch.zeros([x.shape[0], 4])
        return torch.hstack([state_dot, input_dot])

# need sample the mujoco

def random_state(env_bounding_box=4.0, init_max_attitude=np.pi/3.0, init_max_vel=0.5, init_max_angular_vel=0.1*np.pi):
    desired_position = np.zeros(3)
    # attitude (roll pitch yaw)
    quat_init = np.array([1., 0., 0., 0.])
    attitude_euler_rand = np.random.uniform(low=-init_max_attitude, high=init_max_attitude, size=(3,))
    quat_init = utils.euler2quat(attitude_euler_rand)
    # position (x, y, z)
    c = 0.2
    ep = np.random.uniform(low=-(env_bounding_box-c), high=(env_bounding_box-c), size=(3,))
    pos_init = ep + desired_position
    # velocity (vx, vy, vz)
    vel_init = utils.sample_unit3d() * init_max_vel
    # angular velocity (wx, wy, wz)
    angular_vel_init = utils.sample_unit3d() * init_max_angular_vel
    # omegas 
    omegas_init = np.array([params['w_hover']]*4)
    state_init = np.concatenate([pos_init, quat_init, vel_init, angular_vel_init, omegas_init]).ravel()
    return state_init

def simulate(quad_mj, nsim):
    # u = (np.random.uniform([0,0,0,0]) - 0.5)*1e-5 # between -0.25, 0.25
    x = random_state()
    quad_mj.mj_reset(x)
    u_history = []
    for i in range(nsim):
        u = (np.random.uniform([0,0,0,0]) - 0.5)*1e-3 # between -0.25, 0.25
        quad_mj.step(u)
        u_history.append(u)

    history = np.hstack([np.vstack(quad_mj.state_history), np.vstack(u_history)])
    state_dot_history = np.diff(history, axis=0)
    return state_dot_history
    
def get_data(quad_mj, nsim, nsteps, bs):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.ODE_NonAutonomous)
    :param normalize: (bool) Whether to normalize the data
    """
    train_sim, dev_sim, test_sim = [simulate(quad_mj, nsim=nsim) for i in range(3)]
    nx = 21
    nbatch = nsim//nsteps
    length = (nsim//nsteps) * nsteps

    trainX = train_sim[:length].reshape(nbatch, nsteps, nx)
    trainX = torch.tensor(trainX, dtype=torch.float32)
    train_data = DictDataset({'X': trainX, 'xn': trainX[:, 0:1, :]}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                              collate_fn=train_data.collate_fn, shuffle=True)
    
    devX = dev_sim[:length].reshape(nbatch, nsteps, nx)
    devX = torch.tensor(devX, dtype=torch.float32)
    dev_data = DictDataset({'X': devX, 'xn': devX[:, 0:1, :]}, name='dev')
    dev_loader = DataLoader(dev_data, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=True)

    testX = test_sim[:length].reshape(1, nbatch * nsteps, nx)
    testX = torch.tensor(testX, dtype=torch.float32)
    test_data = {'X': testX, 'xn': testX[:, 0:1, :]}

    return train_loader, dev_loader, test_data


if __name__ == '__main__':

    Ts = 0.1

    # Sample rollout of true system
    # -----------------------------
    state = params['default_init_state_np']
    reference = waypoint_reference('wp_p2p', average_vel=1.0)
    quad_mj = QuadcopterMJ(state=state, reference=reference)
    train_loader, dev_loader, test_data = get_data(quad_mj, nsim=10000, nsteps=3, bs=32)

    # Define learning system
    # ----------------------

    # prior is the equations of motion
    quad_nm = QuadcopterNM()
    quad_learner = QuadcopterParam(state_dot=quad_nm.forward)
    fxEul = integrators.Euler(quad_learner, h=Ts)
    dynamics_model = System([Node(fxEul, ['xn'], ['xn'])])

    # Constraints and Losses
    # ----------------------
    x = variable('X')
    xhat = variable('xn')[:, :-1, :]

    xFD = (x[:, 1:, :] - x[:, :-1, :])
    xhatFD = (xhat[:, 1:, :] - xhat[:, :-1, :])

    fd_loss = 2.0*((xFD == xhatFD)^2)
    fd_loss.name = 'FD_loss'

    reference_loss = ((xhat == x)^2)
    reference_loss.name = "ref_loss"

    objectives = [reference_loss, fd_loss]
    constraints = []
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem([dynamics_model], loss)
    # plot computational graph

    # Setup Training
    # --------------
    epochs = 200
    optimizer = torch.optim.Adam(problem.parameters(), lr=0.1)
    logger = BasicLogger(args=None, savedir='test', verbosity=1,
                         stdout=['dev_loss', 'train_loss'])

    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_data,
        optimizer,
        patience=10,
        warmup=10,
        epochs=epochs,
        eval_metric="dev_loss",
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="dev_loss",
        logger=logger,
    )

    # Train
    # -----

    best_model = trainer.train()
    problem.load_state_dict(best_model)

    # dynamics_model.nodes[0].callable(state)
    learned_state_dot = dynamics_model.nodes[0].callable
    
    state0 = test_data['xn'].squeeze().squeeze().unsqueeze(0)

    # Test set results
    test_outputs = dynamics_model(test_data)

    pred_traj = test_outputs['xn'][:, :-1, :]
    true_traj = test_data['X']
    pred_traj = pred_traj.detach().numpy().reshape(-1, 21)
    true_traj = true_traj.detach().numpy().reshape(-1, 21)
    pred_traj, true_traj = pred_traj.transpose(1, 0), true_traj.transpose(1, 0)

    # Create a figure and subplots
    fig, axs = plt.subplots(17, 1, figsize=(10, 40), sharex=True)

    # Plot each dimension in subplots
    for i in range(17):
        axs[i].plot(pred_traj[i], label=f'Set 1 Dimension {i+1}')
        axs[i].plot(true_traj[i], label=f'Set 2 Dimension {i+1}')
        axs[i].set_ylabel('Values')
        axs[i].legend()

    # Set common x-axis label and title
    axs[-1].set_xlabel('Timesteps')
    fig.suptitle('Timeseries Data Comparison')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

    # sample env (eom or mj), train quad, train control policy given quad model
    print('fin')



