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
from dpc_sf.control.dpc.operations import Dynamics, processP2PPolicyInput, processFig8TrajPolicyInput, processP2PTrajPolicyInput, radMultiplier, StateSeqTransformer, MemSeqTransformer, processP2PMemSeqPolicyInput

# torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
np.random.seed(0)
ptu.init_gpu(use_gpu=False)

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

process_policy_input = processFig8TrajPolicyInput(use_error=args["fig8_observe_error"])
process_policy_input_node = Node(process_policy_input, ['X', 'R', 'P'], ['Obs'], name='preprocess')
policy_insize = nx + nr * (not args["fig8_observe_error"]) + 4 # state, reference, equation parameters
node_list.append(process_policy_input_node)

policy = blocks.MLP(
    insize=policy_insize, outsize=nu, bias=True,
    linear_map=torch.nn.Linear,
    nonlin=torch.nn.ReLU,
    hsizes=[20, 20, 20, 20]
).to(ptu.device)
policy_node = Node(policy, ['Obs'], ['U'], name='policy')
node_list.append(policy_node)

dynamics = Dynamics(insize=9, outsize=6, x_std=args["x_noise_std"])
integrator = integrators.Euler(dynamics, h=torch.tensor(args["Ts"]))
dynamics_node = Node(integrator, ['X', 'U'], ['X'], name='dynamics')
node_list.append(dynamics_node)

print(f'node list used in cl_system: {node_list}')
# node_list = [node.callable.to(ptu.device) for node in node_list]
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
    average_velocity        = args["fig8_average_velocity"],
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

action_loss = args["R"] * (u == ptu.tensor(0.))^2  # control penalty
action_loss.name = 'action_loss'
objectives.append(action_loss)

pos_loss = args["Qpos"] * (x[:,:,::2] == r[:,:,::2])^2
pos_loss.name = 'pos_loss'
objectives.append(pos_loss)

vel_loss = args["Qvel"] * (x[:,:,1::2] == r[:,:,1::2])^2
vel_loss.name = 'vel_loss'
objectives.append(vel_loss)

# objectives = [action_loss, pos_loss, vel_loss]
loss = BarrierLoss(objectives, constraints, barrier=args["barrier_type"], alpha=args["barrier_alpha"])
optimizer = torch.optim.Adagrad(policy.parameters(), lr=args["lr"])
problem = Problem([cl_system], loss, grad_inference=True)

# Custom Callack Setup
# --------------------
callback = SinTrajCallback(save_dir=current_datetime, media_path=args["media_path"], nstep=args["nstep"], nx=nx, Ts=args["Ts"])

# Perform the Training
# --------------------
for i in range(args["iterations"]):
    print(f'training with prediction horizon: {args["nstep"]}, lr: {args["lr"]}, delta_terminal: {args["delta_terminal"]}')

    # Get First Datasets
    # ------------
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
        lr_scheduler=False,
        device=ptu.device
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