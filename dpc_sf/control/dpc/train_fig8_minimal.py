from datetime import datetime
import os
import torch 
import numpy as np

from neuromancer.system import Node, System
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.loss import BarrierLoss
from neuromancer.modules import blocks
from neuromancer.dynamics import integrators

# replace this for supercomputer training john
from dpc_sf.utils import pytorch_utils as ptu
from dpc_sf.control.dpc.callback import SinTrajCallback
from dpc_sf.control.dpc.generate_dataset import DatasetGenerator

# call the argparser to get parameters for run
from dpc_sf.control.dpc.operations import Dynamics, processFig8TrajPolicyInput

def train_fig8(
        radius          = 0.50,                      # cylinder radius
        nstep           = 100,                       # number of steps per rollout
        epochs          = 10,                        # number of times we train on the dataset
        iterations      = 1,                         # number of times we retrain on a new dataset "epoch" times
        lr              = 0.05,                      # learning rate
        Ts              = 0.1,                       # timestep of surrogate high level simulation
        minibatch_size  = 10,                        # number of rollouts per gradient step during training
        batch_size      = 5000,                      # number of rollouts in dataset and per epoch
        x_range         = 3.,                        # range of x generated in dataset
        r_range         = 3.,                        # range of r generated in dataset
        cyl_range       = 3.,                        # range of cylinder positions in dataset
        lr_multiplier   = 0.2,                       # in subsequent iterations multiply lr by this number
        sample_type     = 'uniform',                 # 
        barrier_type    = 'softexp',                 # the constraint violations are penalised using a soft exponential function
        barrier_alpha   = 0.05,                      # the decay parameter for the constraint violation parameter away from constraint
        Qpos            = 5.00,                      # position error penalty
        Qvel            = 5.00,                      # velocity error penalty 
        R               = 0.1,                       # input penalty
        optimizer       = 'adagrad',                 #
        fig8_observe_error = True,
        p2p_dataset     = 'cylinder_random',         #
        save_path       = "data/policy/DPC_fig8/",    #
        media_path      = "data/media/dpc/images/"  #
    ):
    # save hyperparameters used
    # -------------------------
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    images_path = media_path + current_datetime + '/'

    # Check if directory exists, and if not, create it
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # NeuroMANCER System Definition
    # -----------------------------

    nx = 6 # state size
    nr = 6 # reference size
    nu = 3 # input size

    # Variables:
    r = variable('R')           # the reference
    u = variable('U')           # the input
    x = variable('X')           # the state
    P = variable('P')           # the equation parameters of the fig8

    # Nodes:
    node_list = []

    process_policy_input = processFig8TrajPolicyInput(use_error=fig8_observe_error)
    process_policy_input_node = Node(process_policy_input, ['X', 'R', 'P'], ['Obs'], name='preprocess')
    policy_insize = nx + nr * (not fig8_observe_error) + 4 # state, reference, equation parameters
    node_list.append(process_policy_input_node)

    policy = blocks.MLP(
        insize=policy_insize, outsize=nu, bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ReLU,
        hsizes=[20, 20, 20, 20]
    ).to(ptu.device)
    policy_node = Node(policy, ['Obs'], ['U'], name='policy')
    node_list.append(policy_node)

    dynamics = Dynamics(insize=9, outsize=6)
    integrator = integrators.Euler(dynamics, h=torch.tensor(Ts))
    dynamics_node = Node(integrator, ['X', 'U'], ['X'], name='dynamics')
    node_list.append(dynamics_node)

    print(f'node list used in cl_system: {node_list}')
    # node_list = [node.callable.to(ptu.device) for node in node_list]
    cl_system = System(node_list)

    print(batch_size)

    # Dataset Generation Class
    # ------------------------
    dataset = DatasetGenerator(
        p2p_dataset             = p2p_dataset,
        task                    = 'fig8',
        x_range                 = x_range,
        r_range                 = r_range,
        cyl_range               = cyl_range,
        radius                  = radius,
        batch_size              = batch_size,
        minibatch_size          = minibatch_size,
        nstep                   = nstep,
        sample_type             = sample_type,
        nx                      = nx,
        validate_data           = True,
        shuffle_dataloaders     = False,
        average_velocity        = 1.0,
        device                  = ptu.device
    )


    # Optimisation Problem Setup
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

    # Define Constraints:
    constraints = []

    # Define Loss:
    objectives = []

    action_loss = R * (u == ptu.tensor(0.))^2  # control penalty
    action_loss.name = 'action_loss'
    objectives.append(action_loss)

    pos_loss = Qpos * (x[:,:,::2] == r[:,:,::2])^2
    pos_loss.name = 'pos_loss'
    objectives.append(pos_loss)

    vel_loss = Qvel * (x[:,:,1::2] == r[:,:,1::2])^2
    vel_loss.name = 'vel_loss'
    objectives.append(vel_loss)

    # objectives = [action_loss, pos_loss, vel_loss]
    loss = BarrierLoss(objectives, constraints, barrier=barrier_type, alpha=barrier_alpha)
    optimizer = torch.optim.Adagrad(policy.parameters(), lr=lr)
    problem = Problem([cl_system], loss, grad_inference=True)

    # Custom Callack Setup
    # --------------------
    callback = SinTrajCallback(save_dir=current_datetime, media_path=media_path, nstep=nstep, nx=nx, Ts=Ts)

    # Perform the Training
    # --------------------
    for i in range(iterations):
        print(f'training with prediction horizon: {nstep}, lr: {lr}')

        # Get First Datasets
        # ------------
        dataset.nstep = nstep
        train_loader, dev_loader = dataset.get_loaders()

        trainer = Trainer(
            problem,
            train_loader,
            dev_loader,
            dev_loader,
            optimizer,
            callback=callback,
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


    callback.animate()
    callback.delete_all_but_last_image()

    # Save the Policy
    # ---------------

    # %%
    policy_state_dict = {}
    for key, value in best_model.items():
        if "callable." in key:
            new_key = key.split("nodes.0.nodes.1.")[-1]
            policy_state_dict[new_key] = value
    torch.save(policy_state_dict, save_path + f"policy.pth")

if __name__ == "__main__":
    
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    np.random.seed(0)
    ptu.init_gpu(use_gpu=True)

    train_fig8()