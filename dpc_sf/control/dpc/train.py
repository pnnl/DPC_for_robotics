from datetime import datetime
import os
import json 
import torch 
import numpy as np

from neuromancer.system import Node, System
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.loss import BarrierLoss
from neuromancer.modules import blocks
from neuromancer.dynamics import integrators, ode
from neuromancer.callbacks import Callback

# replace this for supercomputer training john
from dpc_sf.utils import pytorch_utils as ptu
from dpc_sf.control.dpc.callback import WP_Callback, LinTrajCallback, SinTrajCallback
from dpc_sf.control.dpc.generate_dataset import DatasetGenerator

# call the argparser to get parameters for run
from dpc_sf.control.dpc.train_params import args_dict as args
from dpc_sf.control.dpc.operations import posVel2cyl, Dynamics, processP2PPolicyInput, processFig8TrajPolicyInput, processP2PTrajPolicyInput, radMultiplier, BimodalPolicy

# torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
np.random.seed(0)

# save hyperparameters used
# -------------------------
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
images_path = args["media_path"] + current_datetime + '/'

# Check if directory exists, and if not, create it
if not os.path.exists(images_path):
    os.makedirs(images_path)

# Save to JSON
with open(images_path + 'args.json', 'w') as f:
    json.dump(args, f, indent=4)

# NeuroMANCER System Definition
# -----------------------------

nx = 6 # state size
nr = 6 # reference size
nu = 3 # input size
nc = 2 # cylinder distance and velocity

# Variables:
r = variable('R')           # the reference
u = variable('U')           # the input
x = variable('X')           # the state
idx = variable('Idx')       # the index of the current timestep into the horizon, used to find m below.

if args["task"] == 'wp_p2p':
    cyl = variable('Cyl')       # the cylinder center coordinates
    m = variable('M')           # radius multiplier, increase radius into horizon future
elif args["task"] == 'fig8':
    P = variable('P')   # the equation parameters of the fig8

# Nodes:
node_list = []

if args["use_rad_multiplier"] is True:
    rad_multiplier = radMultiplier(Ts=args["Ts"], bs=args["batch_size"])
    rad_multiplier_node = Node(rad_multiplier, ['Idx'], ['Idx', 'M'], name='rad_multiplier')
    node_list.append(rad_multiplier_node)

if args["task"] == 'wp_p2p':
    process_policy_input = processP2PPolicyInput(radius=args["radius"])
    process_policy_input_node = Node(process_policy_input, ['X', 'R', 'Cyl'], ['Obs'], name='preprocess')
    policy_insize = nx + nc
elif args["task"] == 'fig8':
    process_policy_input = processFig8TrajPolicyInput(use_error=args["fig8_observe_error"])
    process_policy_input_node = Node(process_policy_input, ['X', 'R', 'P'], ['Obs'], name='preprocess')
    policy_insize = nx + nr * (not args["fig8_observe_error"]) + 4 # state, reference, equation parameters
elif args["task"] == 'wp_traj':
    process_policy_input = processP2PTrajPolicyInput(use_error=True)
    process_policy_input_node = Node(process_policy_input, ['X', 'R'], ['Obs'], name='preprocess')
    policy_insize = nx
node_list.append(process_policy_input_node)

if (args["p2p_bimodal_policy"] is True) and (args["task"] == 'wp_p2p'):
    policy = BimodalPolicy()
else:
    policy = blocks.MLP(
        insize=policy_insize, outsize=nu, bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ReLU,
        hsizes=[20, 20, 20, 20]
    )
policy_node = Node(policy, ['Obs'], ['U'], name='policy')
node_list.append(policy_node)

dynamics = Dynamics(insize=9, outsize=6, x_std=args["x_noise_std"])
integrator = integrators.Euler(dynamics, h=torch.tensor(args["Ts"]))
dynamics_node = Node(integrator, ['X', 'U'], ['X'], name='dynamics')
node_list.append(dynamics_node)

print(f'node list used in cl_system: {node_list}')
cl_system = System(node_list)

print(args['batch_size'])

# Dataset Generation Class
# ------------------------
dataset = DatasetGenerator(
    p2p_dataset             = args["p2p_dataset"],
    task                    = args["task"],
    x_range                 = args["x_range"],
    r_range                 = args["r_range"],
    cyl_range               = args["cyl_range"],
    radius                  = args["radius"],
    batch_size              = args["batch_size"],
    minibatch_size          = args["minibatch_size"],
    nstep                   = args["nstep"],
    sample_type             = args["sample_type"],
    nx                      = nx,
    validate_data           = args["validate_data"],
    shuffle_dataloaders     = args["shuffle_dataloaders"],
    average_velocity        = args["fig8_average_velocity"]
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
if args["use_cyl_constraint"] is True and (args["task"] == 'wp_p2p'):
    cylinder_constraint = args["Q_con"] * ((args["radius"]**2 * m[:,:,0] <= (x[:,:,0]-cyl[:,:,0])**2 + (x[:,:,2]-cyl[:,:,1])**2)) ^ 2
    constraints.append(cylinder_constraint)

if args["use_terminal_state_penalty"] is True:
    terminal_lower_bound_penalty = args["Q_terminal"] * (x[:,-1,:] > r[:,-1,:] - args["delta_terminal"])
    terminal_upper_bound_penalty = args["Q_terminal"] * (x[:,-1,:] < r[:,-1,:] + args["delta_terminal"])
    constraints.append(terminal_lower_bound_penalty)
    constraints.append(terminal_upper_bound_penalty)

# Define Loss:
objectives = []

action_loss = args["R"] * (u == 0.)^2  # control penalty
action_loss.name = 'action_loss'
objectives.append(action_loss)

pos_loss = args["Qpos"] * (x[:,:,::2] == r[:,:,::2])^2
pos_loss.name = 'pos_loss'
objectives.append(pos_loss)

if args["use_terminal_state_penalty"] is True:
    term_pos_loss = args["Qpos"] * args["Qtermscale"] * (x[:,-1,::2] == r[:,-1,::2])^2
    term_pos_loss.name = 'term_pos_loss'
    objectives.append(term_pos_loss)

    term_vel_loss = args["Qvel"] * args["Qtermscale"] * (x[:,-1,1::2] == r[:,-1,1::2])^2
    term_vel_loss.name = 'term_vel_loss'
    objectives.append(term_vel_loss)

vel_loss = args["Qvel"] * (x[:,:,1::2] == r[:,:,1::2])^2
vel_loss.name = 'vel_loss'
objectives.append(vel_loss)

# objectives = [action_loss, pos_loss, vel_loss]
loss = BarrierLoss(objectives, constraints, barrier=args["barrier_type"], alpha=args["barrier_alpha"])

# Define the Problem and the Trainer:
problem = Problem([cl_system], loss, grad_inference=True)
if args["optimizer"] == 'adamw':
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args["lr"])
elif args["optimizer"] == 'adagrad':
    optimizer = torch.optim.Adagrad(policy.parameters(), lr=args["lr"])
else:
    raise Exception

# Custom Callack Setup
# --------------------
if args["use_custom_callback"] is True and (args["task"] == 'wp_p2p'):
    callback = WP_Callback(save_dir=current_datetime, media_path=args["media_path"], nstep=args["nstep"], nx=nx)
elif args["use_custom_callback"] is True and (args["task"] == 'wp_traj'):
    callback = LinTrajCallback(save_dir=current_datetime, media_path=args["media_path"], nstep=args["nstep"], nx=nx, Ts=args["Ts"])
elif args["use_custom_callback"] is True and (args["task"] == 'fig8'):
    callback = SinTrajCallback(save_dir=current_datetime, media_path=args["media_path"], nstep=args["nstep"], nx=nx, Ts=args["Ts"])
else:
    callback = Callback()

# Perform the Training
# --------------------
if args["use_old_datasets"] is True:
    train_loader, dev_loader = dataset.get_loaders()

if args["train"] is True:
    # Train model with prediction horizon of train_nsteps
    for i in range(args["iterations"]):
        print(f'training with prediction horizon: {args["nstep"]}, lr: {args["lr"]}, delta_terminal: {args["delta_terminal"]}')

        # Get First Datasets
        # ------------
        if args["use_old_datasets"] is False:
            dataset.nstep = args["nstep"]
            train_loader, dev_loader = dataset.get_loaders()

        trainer = Trainer(
            problem,
            train_loader,
            dev_loader,
            dev_loader,
            optimizer,
            callback=callback,
            epochs=args["epochs"],
            patience=args["epochs"],
            train_metric="train_loss",
            dev_metric="dev_loss",
            test_metric="test_loss",
            eval_metric='dev_loss',
            warmup=400,
            lr_scheduler=False
        )

        # Train Over Nsteps
        # -----------------
        cl_system.nsteps = args["nstep"]
        best_model = trainer.train()
        trainer.model.load_state_dict(best_model)

        # Update Parameters for the Next Iteration
        # ----------------------------------------
        args["lr"]              *= args["lr_multiplier"] # 0.2
        args["Q_con"]           *= 1
        args["Q_terminal"]      *= 1
        args["delta_terminal"]  *= 1 # 0.2
        args["nstep"]           *= args["nstep_multiplier"] # 1

        # update the prediction horizon
        cl_system.nsteps = args["nstep"]

        if args["use_old_datasets"] is True:
            # Get New Datasets
            # ----------------
            train_data, dev_data = dataset.get_dictdatasets()

            # apply new training data and learning rate to trainer
            trainer.train_data, trainer.dev_data, trainer.test_data = train_data, dev_data, dev_data

        optimizer.param_groups[0]['lr'] = args["lr"]

else:

    best_model = torch.load(args["save_path"] + "policy.pth")

if args["use_custom_callback"] is True:
    callback.animate()
    callback.delete_all_but_last_image()

# Save the Policy
# ---------------

# %%
if args["train"] is True:
    policy_state_dict = {}
    for key, value in best_model.items():
        if "callable." in key:
            new_key = key.split("nodes.0.nodes.1.")[-1]
            policy_state_dict[new_key] = value
    torch.save(policy_state_dict, args["save_path"] + f"policy_experimental_tt.pth")
else:
    print(f"not saving policy as train is: {args['train']}")