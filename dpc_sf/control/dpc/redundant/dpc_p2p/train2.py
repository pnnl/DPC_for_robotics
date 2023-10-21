# # Waypoint based obstacle avoidance DPC
# 
# I call it wp_p2p which stands for waypoint point to point, but the title is more accurate ^

from datetime import datetime
import os
import imageio
import argparse
import json 

import torch 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from neuromancer.system import Node, System
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import BarrierLoss
from neuromancer.modules import blocks
from neuromancer.plot import pltCL
from neuromancer.dynamics import integrators, ode
from neuromancer.callbacks import Callback

# replacing this dependancy so that you don't need to install my repo.
# from dpc_sf.utils import pytorch_utils as ptu
class Ptu:
    def __init__(self) -> None:
        
        self.device = None
        self.dtype = torch.float32

    def init_gpu(self, use_gpu=True, gpu_id=0):
        global device
        if torch.cuda.is_available() and use_gpu:
            device = torch.device("cuda:" + str(gpu_id))
            print("Using GPU id {}".format(gpu_id))
        else:
            device = torch.device("cpu")
            print("GPU not detected. Defaulting to CPU.")

    def init_dtype(self, set_dtype=torch.float32):
        global dtype
        dtype = set_dtype
        print(f"Use dtype: {dtype}")

    def set_device(self, gpu_id):
        torch.cuda.set_device(gpu_id)

    def from_numpy(self, *args, **kwargs):
        return torch.from_numpy(*args, **kwargs).type(self.dtype).to(self.device)

    def to_numpy(self, tensor):
        return tensor.to('cpu').detach().numpy()

    def create_tensor(self, list):
        return torch.tensor(list).type(self.dtype).to(self.device)

    def create_zeros(self, shape):
        return torch.zeros(shape).type(self.dtype).to(self.device)
ptu = Ptu()

# torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)

# ## Options

# Create the parser
parser = argparse.ArgumentParser(description='Command line options for your program')

# Add arguments
parser.add_argument('--radius',                     type=float, default=0.50,                     help='Radius of the cylinder to be avoided')
parser.add_argument('--save_path',                  type=str,   default="data/policy/DPC_p2p/",   help='Path to save the policies')
parser.add_argument('--media_path',                 type=str,   default="data/media/dpc/images/", help='Path to save the policies')
parser.add_argument('--nstep',                      type=int,   default=100,                      help='Number of timesteps in the horizon')
parser.add_argument('--epochs',                     type=int,   default=30,                       help='Number of training epochs per iteration')
parser.add_argument('--iterations',                 type=int,   default=2,                        help='Number of iterations')
parser.add_argument('--lr',                         type=float, default=0.01,                     help='Learning rate')
parser.add_argument('--Ts',                         type=float, default=0.1,                      help='Timestep')
parser.add_argument('--minibatch_size',             type=int,   default=500,                       help='Autograd per minibatch size')
parser.add_argument('--batch_size',                 type=int,   default=5000,                      help='Batch size')
parser.add_argument('--x_range',                    type=float, default=6.,                       help='Multiplier for the initial states')
parser.add_argument('--r_range',                    type=float, default=6.,                       help='Multiplier for the reference end points')
parser.add_argument('--cyl_range',                  type=float, default=6.,                       help='Multiplier for the cylinder center location')
parser.add_argument('--use_rad_multiplier',         type=bool,  default=False,                    help='Increase radius over the horizon')
parser.add_argument('--train',                      type=bool,  default=True,                     help='Train the model')
parser.add_argument('--use_terminal_state_penalty', type=bool,  default=False,                    help='Use terminal state penalty in the cost function')
parser.add_argument('--Q_con',                      type=float, default=1_000_000.,               help='Cost of violating the cylinder constraint')
parser.add_argument('--Q_terminal',                 type=float, default=1.0,                      help='Cost of violating the terminal constraint')
parser.add_argument('--delta_terminal',             type=float, default=0.1,                      help='Terminal constraint initial radius')
parser.add_argument('--x_noise_std',                type=float, default=0.0,                      help='Terminal constraint initial radius')
parser.add_argument('--use_custom_callback',        type=bool,  default=True,                      help='Terminal constraint initial radius')
parser.add_argument('--lr_multiplier',              type=float, default=0.2,                      help='Terminal constraint initial radius')
parser.add_argument('--nstep_multiplier',           type=float, default=1,                      help='Terminal constraint initial radius')
parser.add_argument('--use_old_datasets',           type=bool,  default=False,                      help='Terminal constraint initial radius')
parser.add_argument('--sample_type',                type=str,  default='uniform',                      help='Terminal constraint initial radius')
parser.add_argument('--barrier_type',                type=str,  default='softexp',                      help='Terminal constraint initial radius')
parser.add_argument('--barrier_alpha',                type=float,  default=0.3,                      help='Terminal constraint initial radius')


# Parse the arguments
args = parser.parse_args()

radius                      = args.radius                    
save_path                   = args.save_path                 
media_path                  = args.media_path                 
nstep                       = args.nstep                     
epochs                      = args.epochs                    
iterations                  = args.iterations                
lr                          = args.lr                        
Ts                          = args.Ts                        
minibatch_size              = args.minibatch_size            
batch_size                  = args.batch_size                
x_range                     = args.x_range                   
r_range                     = args.r_range                   
cyl_range                   = args.cyl_range                 
use_rad_multiplier          = args.use_rad_multiplier        
train                       = args.train                     
use_terminal_state_penalty  = args.use_terminal_state_penalty
Q_con                       = args.Q_con                     
Q_terminal                  = args.Q_terminal                
delta_terminal              = args.delta_terminal   
x_noise_std                 = args.x_noise_std         
use_custom_callback         = args.use_custom_callback         
lr_multiplier               = args.lr_multiplier         
nstep_multiplier            = args.nstep_multiplier    
use_old_datasets            = args.use_old_datasets
sample_type                 = args.sample_type # 'uniform', 'normal'
barrier_type                 = args.barrier_type # 'uniform', 'normal'
barrier_alpha                 = args.barrier_alpha # 'uniform', 'normal'

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
images_path = media_path + current_datetime + '/'
# Convert args namespace to dictionary
args_dict = vars(args)

# Check if directory exists, and if not, create it
if not os.path.exists(images_path):
    os.makedirs(images_path)

# Save to JSON
with open(images_path + 'args.json', 'w') as f:
    json.dump(args_dict, f, indent=4)

# ## Triple Double Integrator Definition

nx = 6 # state size
nr = 6 # reference size
nu = 3 # input size
nc = 2 # cylinder distance and velocity

interp_u = lambda tq, t, u: u

A = torch.tensor([
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])

B = torch.tensor([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0]
])

# variables
cyl = variable('Cyl')       # the cylinder center coordinates
r = variable('R')           # the reference
u = variable('U')           # the input
x = variable('X')           # the state
idx = variable('Idx')       # the index of the current timestep into the horizon, used to find m below.
m = variable('M')           # radius multiplier, increase radius into horizon future
I_err = variable('I_err')   # the integrated error term used by the integrator policy

# Classes:
class Dynamics(ode.ODESystem):
    def __init__(self, insize, outsize, x_std=0.0) -> None:
        super().__init__(insize=insize, outsize=outsize)
        self.f = lambda x, u: x @ A.T + u @ B.T
        self.in_features = insize
        self.out_features = outsize
        
        # noise definitions
        self.x_std = x_std

    def ode_equations(self, xu):
        x = xu[:,0:6]
        u = xu[:,6:9]
        # add noise if required
        x = x + torch.randn(x.shape) * self.x_std
        return self.f(x,u)

class Cat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args):
        # expects shape [bs, nx]
        return torch.hstack(args)
    
def posVel2cyl(state, cyl, radius=radius):
    x = state[:, 0:1]
    y = state[:, 2:3]
    xc = cyl[:, 0:1]
    yc = cyl[:, 1:2]

    dx = x - xc
    dy = y - yc

    # Calculate the Euclidean distance from each point to the center of the cylinder
    distance_to_center = (dx**2 + dy**2) ** 0.5
    
    # Subtract the radius to get the distance to the cylinder surface
    distance_to_cylinder = distance_to_center - radius

    xdot = state[:, 1:2]
    ydot = state[:, 3:4]

    # Normalize the direction vector (from the point to the center of the cylinder)
    dx_normalized = dx / (distance_to_center + 1e-10)  # Adding a small number to prevent division by zero
    dy_normalized = dy / (distance_to_center + 1e-10)

    # Compute the dot product of the normalized direction vector with the velocity vector
    velocity_to_cylinder = dx_normalized * xdot + dy_normalized * ydot

    return distance_to_cylinder, velocity_to_cylinder


class processPolicyInput(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, r, cyl):
        # we want to pass the 
        e = r - x
        c_pos, c_vel = posVel2cyl(x, cyl)
        return torch.hstack([e, c_pos, c_vel])



class radMultiplier(torch.nn.Module):
    def __init__(self, Ts, bs=1) -> None:
        super().__init__()
        self.Ts = Ts
        self.bs = bs

    def forward(self, i):
        # multiplier = 1 + idx * Ts * 0.5
        m = 1 + i * self.Ts * 0.01 # 0.15 # 0.5
        i = i + 1
        return i, m

# Nodes:
node_list = []

if use_rad_multiplier:

    rad_multiplier = radMultiplier(Ts=Ts, bs=batch_size)
    rad_multiplier_node = Node(rad_multiplier, ['Idx'], ['Idx', 'M'], name='rad_multiplier')
    node_list.append(rad_multiplier_node)

# state_ref_cat = Cat()
# state_ref_cat_node = Node(state_ref_cat, ['X', 'R', 'Cyl'], ['XRC'], name='cat')
# node_list.append(state_ref_cat_node)

process_policy_input = processPolicyInput()
process_policy_input_node = Node(process_policy_input, ['X', 'R', 'Cyl'], ['XRC'], name='preprocess')
node_list.append(process_policy_input_node)

policy = blocks.MLP(
    insize=nx + nc, outsize=nu, bias=True,
    linear_map=torch.nn.Linear,
    nonlin=torch.nn.ReLU,
    hsizes=[20, 20, 20, 20]
)
policy_node = Node(policy, ['XRC'], ['U'], name='policy')
node_list.append(policy_node)

dynamics = Dynamics(insize=9, outsize=6, x_std=x_noise_std)
integrator = integrators.Euler(dynamics, interp_u=interp_u, h=torch.tensor(Ts))
dynamics_node = Node(integrator, ['X', 'U'], ['X'], name='dynamics')
node_list.append(dynamics_node)

print(f'node list used in cl_system: {node_list}')
cl_system = System(node_list)
# cl_system.show() # I cannot run the .show() in the notebook (pydot issues), but I can run in debug mode?

# ## Dataset Generation
# 
# Algorithm for dataset generation:
# 
# - we randomly sample 3 points for each rollout, quad start state, quad reference state, cylinder position
# - if a particular datapoint has a start or reference state that is contained within the random cylinder 
#   we discard that datapoint and try again. The output of this is a "filtered" dataset in "get_filtered_dataset"
# - I also have a validation function to check the filtered datasets produced here, but you can ignore that
# - these filtered datasets are then wrapped in DictDatasets and then wrapped in torch dataloaders
# 
# NOTE:
# - the minibatch size used in the torch dataloader can play a key role in reducing the final steady state error
#   of the system
#     - If the minibatch is too large we will not get minimal steady state error, minibatch of 10 has proven good

# average end position, for if we want to go to a certain location more than others, 0. if not
end_pos = ptu.create_tensor([[[0.0,0.0,0.0,0.0,0.0,0.0]]])

def is_inside_cylinder(x, y, cx, cy, radius=radius):
    """
    Check if a point (x,y) is inside a cylinder with center (cx, cy) and given radius.
    """
    distance = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return distance < radius

def get_random_state(x_range, r_range, cyl_range, sample_type='uniform'):
    if sample_type == 'normal':
        x_sample = x_range * torch.randn(1, 1, nx)
        r_sample = torch.cat([r_range * torch.randn(1, 1, nx)] * (nstep + 1), dim=1) + end_pos
        cyl_sample = torch.cat([cyl_range * torch.randn(1, 1, 2)] * (nstep + 1), dim=1)
    elif sample_type == 'uniform':
        x_sample = 2 * x_range * (torch.rand(1, 1, nx) - 0.5)
        r_sample = torch.cat([2 * r_range * (torch.rand(1, 1, nx) - 0.5)] * (nstep + 1), dim=1) + end_pos
        cyl_sample = torch.cat([2 * cyl_range * (torch.rand(1, 1, 2) - 0.5)] * (nstep + 1), dim=1)
    else:
        raise ValueError(f"invalid sample type passed: {sample_type}")
    # reference velocities should be zero here 
    r_sample[:,:,1::2] = 0.
    return x_sample, r_sample, cyl_sample

def get_filtered_dataset(batch_size, nx, nstep, x_range, r_range, end_pos, cyl_range):    
    """
    Generate a filtered dataset of samples where none of the state or reference points are inside a cylinder.

    Parameters:
    - batch_size (int): Desired number of data samples in the batch.
    - nx (int): Dimension of the state sample.
    - nstep (int): Number of steps for reference and cylinder data.
    - x_range (float): Scaling factor for state data.
    - r_range (float): Scaling factor for reference data.
    - end_pos (float): Constant offset for reference data.
    - cyl_range (float): Scaling factor for cylinder data.

    Returns:
    - dict: A dictionary containing:
        'X': Tensor of state data of shape [batch_size, 1, nx].
        'R': Tensor of reference data of shape [batch_size, nstep + 1, nx].
        'Cyl': Tensor of cylinder data of shape [batch_size, nstep + 1, 2].
        'Idx': Zero tensor indicating starting index for each sample in batch.
        'M': Tensor with ones indicating starting multiplier for each sample.
        'I_err': Zero tensor indicating initial error for each sample in batch.

    """
    X = []
    R = []
    Cyl = []
    
    # Loop until the desired batch size is reached.
    print(f"generating dataset of batchsize: {batch_size}")
    while len(X) < batch_size:

        x_sample, r_sample, cyl_sample = get_random_state(x_range, r_range, cyl_range, sample_type=sample_type)
        inside_cyl = False
        
        # Check if any state or reference point is inside the cylinder.
        for t in range(nstep + 1):
            if is_inside_cylinder(x_sample[0, 0, 0], x_sample[0, 0, 2], cyl_sample[0, t, 0], cyl_sample[0, t, 1]):
                inside_cyl = True
                break
            if is_inside_cylinder(r_sample[0, t, 0], r_sample[0, t, 2], cyl_sample[0, t, 0], cyl_sample[0, t, 1]):
                inside_cyl = True
                break
        
        if not inside_cyl:
            X.append(x_sample)
            R.append(r_sample)
            Cyl.append(cyl_sample)
    
    # Convert lists to tensors.
    X = torch.cat(X, dim=0)
    R = torch.cat(R, dim=0)
    Cyl = torch.cat(Cyl, dim=0)
    
    return {
        'X': X,
        'R': R,
        'Cyl': Cyl,
        'Idx': ptu.create_zeros([batch_size,1,1]), # start idx
        'M': torch.ones([batch_size, 1, 1]), # start multiplier
        'I_err': ptu.create_zeros([batch_size,1,3])
    }

def validate_dataset(dataset):
    X = dataset['X']
    R = dataset['R']
    Cyl = dataset['Cyl']
    
    batch_size = X.shape[0]
    nstep = R.shape[1] - 1

    print("validating dataset...")
    for i in range(batch_size):
        for t in range(nstep + 1):
            # Check initial state.
            if is_inside_cylinder(X[i, 0, 0], X[i, 0, 2], Cyl[i, t, 0], Cyl[i, t, 1]):
                return False, f"Initial state at index {i} lies inside the cylinder."
            # Check each reference point.
            if is_inside_cylinder(R[i, t, 0], R[i, t, 2], Cyl[i, t, 0], Cyl[i, t, 1]):
                return False, f"Reference at time {t} for batch index {i} lies inside the cylinder."

    return True, "All points are outside the cylinder."

def get_dictdatasets(batch_size, nx, nstep, x_range, r_range, end_pos, cyl_range):

    train_data = DictDataset(get_filtered_dataset(
        batch_size=batch_size,
        nx = nx,
        nstep = nstep,
        x_range = x_range,
        r_range = r_range,
        end_pos = end_pos,
        cyl_range = cyl_range,
    ), name='train')

    dev_data = DictDataset(get_filtered_dataset(
        batch_size=batch_size,
        nx = nx,
        nstep = nstep,
        x_range = x_range,
        r_range = r_range,
        end_pos = end_pos,
        cyl_range = cyl_range,
    ), name='dev')

    return train_data, dev_data

def get_loaders(batch_size, nx, nstep, x_range, r_range, end_pos, cyl_range):

    train_data, dev_data = get_dictdatasets(batch_size, nx, nstep, x_range, r_range, end_pos, cyl_range)

    validate_dataset(train_data.datadict)
    validate_dataset(dev_data.datadict)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=minibatch_size,
                                            collate_fn=train_data.collate_fn, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=minibatch_size,
                                            collate_fn=dev_data.collate_fn, shuffle=False)
    
    return train_loader, dev_loader


## Optimisation Problem Setup

# NOTE:
# - Use of a BarrierLoss, which provides a loss which increases the closer to a constraint violation
#   we come, has proven much more effective than a PenaltyLoss in obstacle avoidance. The PenaltyLoss 
#   is either on or off, and so has a poor learning signal for differentiable problems, but the BarrierLoss
#   provides a constant learning signal which has proven effective.
# - Not penalising velocity, and only position has frequently led to controllers that do not converge
#   to zero velocity, and therefore drift over time. Therefore velocities have been kept in the 
#   regulation loss term.

# Define Constraints
# ------------------
constraints = []
cylinder_constraint = Q_con * ((radius**2 * m[:,:,0] <= (x[:,:,0]-cyl[:,:,0])**2 + (x[:,:,2]-cyl[:,:,1])**2)) ^ 2
constraints.append(cylinder_constraint)

if use_terminal_state_penalty is True:

    terminal_lower_bound_penalty = Q_terminal * (x[:,-1,:] > r[:,-1,:] - delta_terminal)
    terminal_upper_bound_penalty = Q_terminal * (x[:,-1,:] < r[:,-1,:] + delta_terminal)
    constraints.append(terminal_lower_bound_penalty)
    constraints.append(terminal_upper_bound_penalty)

# Define Loss
# -----------
action_loss = 0.1 * (u == 0.)^2  # control penalty
regulation_loss = 5.0 * (x[:,:,:] == r[:,:,:])^2
objectives = [action_loss, regulation_loss]
loss = BarrierLoss(objectives, constraints, barrier=barrier_type, alpha=barrier_alpha)

# Define the Problem and the Trainer
# ----------------------------------
problem = Problem([cl_system], loss, grad_inference=True)
optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)



def plot_traj(trajectories, save_path=media_path):
    # Extract x and y
    x_positions = trajectories['X'][0, :, 0]
    y_positions = trajectories['X'][0, :, 2]
    # Convert the PyTorch tensors to numpy arrays
    x_np = ptu.to_numpy(x_positions)
    y_np = ptu.to_numpy(y_positions)
    fig, ax = plt.subplots()
    # Plot the trajectory
    ax.plot(x_np, y_np)
    # Draw a circle at (1,1) with a radius of 0.5
    circle = Circle((1, 1), 0.5, fill=False)  # fill=False makes the circle hollow. Change to True if you want it filled.
    ax.add_patch(circle)
    # Plot a red dot at the point (2,2)
    ax.scatter(2, 2, color='red')
    # Setting equal scaling and showing the grid:
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    # Save the figure
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    fig.savefig(save_path + f'{current_datetime}.png', dpi=300)  # Change the filename and dpi as needed
    plt.close(fig)  # This closes the figure and releases its resources.


class TSCallback(Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.directory = media_path + save_dir + '/'


    def begin_eval(self, trainer, output):
        # lets call the current trained model on the data we are interested in
        data = {
            'X': torch.zeros(1, 1, nx, dtype=torch.float32), 
            'R': torch.cat([torch.tensor([[[2, 0, 2, 0, 2, 0]]])]*(nstep+1), dim=1), 
            'Cyl': torch.cat([torch.tensor([[[1,1]]])]*(nstep+1), dim=1), 
            'Idx': torch.vstack([torch.tensor([0.0])]).unsqueeze(1),
            'M': torch.ones([1, 1, 1]), # start multiplier
            'I_err': ptu.create_zeros([1,1,3])
        }

        trajectories = trainer.model.nodes[0](data)

        plot_traj(trajectories, save_path=self.directory)
        plt.close()
    
    def animate(self):
        # Gather all the PNG files
        filenames = sorted([f for f in os.listdir(self.directory) if f.endswith('.png')])

        # Convert the PNGs to GIF
        with imageio.get_writer(self.directory + 'animation.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(self.directory + filename)
                writer.append_data(image)

    def delete_all_but_last_image(self):
        # List all .png files in the directory
        all_files = [f for f in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, f)) and f.endswith('.png')]

        # Sort the files (they'll be sorted by date/time due to the filename format)
        sorted_files = sorted(all_files)

        # Delete all but the last one
        for file in sorted_files[:-1]:  # Exclude the last file
            os.remove(os.path.join(self.directory, file))

if use_custom_callback is True:
    
    callback = TSCallback(save_dir=current_datetime)
else:
    callback = Callback()

# %% [markdown]
# ## Training
# 
# NOTE:
# - Multiple iterations with new datasets at lower learning rates has proven very effective at generating
#   superior policies

if use_old_datasets is True:
    train_loader, dev_loader = get_loaders(
        batch_size=batch_size,
        nx=nx,
        nstep=nstep,
        x_range=x_range,
        r_range=r_range,
        end_pos=end_pos,
        cyl_range=cyl_range
    )

# %%
if train is True:
    # Train model with prediction horizon of train_nsteps
    for i in range(iterations):
        print(f'training with prediction horizon: {nstep}, lr: {lr}, delta_terminal: {delta_terminal}')

        # Get First Datasets
        # ------------
        if use_old_datasets is False:
            train_loader, dev_loader = get_loaders(
                batch_size=batch_size,
                nx=nx,
                nstep=nstep,
                x_range=x_range,
                r_range=r_range,
                end_pos=end_pos,
                cyl_range=cyl_range
            )

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
            lr_scheduler=False
        )

        # Train Over Nsteps
        # -----------------
        cl_system.nsteps = nstep            
        best_model = trainer.train()
        trainer.model.load_state_dict(best_model)

        # Update Parameters for the Next Iteration
        # ----------------------------------------
        lr *= lr_multiplier # 0.2
        Q_con *= 1
        Q_terminal *= 1
        delta_terminal *= 1 # 0.2
        nstep *= nstep_multiplier # 1

        # update the prediction horizon
        cl_system.nsteps = nstep

        if use_old_datasets is True:
            # Get New Datasets
            # ------------
            train_data, dev_data = get_dictdatasets(
                batch_size=batch_size,
                nx=nx,
                nstep=nstep,
                x_range=x_range,
                r_range=r_range,
                end_pos=end_pos,
                cyl_range=cyl_range
            )
            # apply new training data and learning rate to trainer
            trainer.train_data, trainer.dev_data, trainer.test_data = train_data, dev_data, dev_data
        optimizer.param_groups[0]['lr'] = lr

else:

    best_model = torch.load(save_path + "policy.pth")

if use_custom_callback is True:
    callback.animate()
    callback.delete_all_but_last_image()

# %% [markdown]
# ## Test Inference

# %%
policy_state_dict = {}
for key, value in best_model.items():
    if "callable." in key:
        if "nodes.0.nodes.1." in key:
            new_key = key.split("nodes.0.nodes.1.")[-1]
            policy_state_dict[new_key] = value
        elif "nodes.0.nodes.2." in key:
            new_key = key.split("nodes.0.nodes.2.")[-1]
            policy_state_dict[new_key] = value
        # new_key = key.split("nodes.0.nodes.1.")[-1]
        # policy_state_dict[new_key] = value
if use_rad_multiplier is True:
    cl_system.nodes[2].load_state_dict(policy_state_dict)
else:
    cl_system.nodes[1].load_state_dict(policy_state_dict)


data = {
    'X': torch.zeros(1, 1, nx, dtype=torch.float32), 
    'R': torch.cat([torch.tensor([[[2, 0, 2, 0, 2, 0]]])]*(nstep+1), dim=1), 
    'Cyl': torch.cat([torch.tensor([[[1,1]]])]*(nstep+1), dim=1), 
    'Idx': torch.vstack([torch.tensor([0.0])]).unsqueeze(1),
    'M': torch.ones([1, 1, 1]), # start multiplier
    'I_err': ptu.create_zeros([1,1,3])
}
cl_system.nsteps = nstep
print(f"testing model over {nstep} timesteps...")

# set system noise to zero
try:
    cl_system.nodes[2].callable.block.x_std = 0
except:
    cl_system.nodes[3].callable.block.x_std = 0


trajectories = cl_system(data)
pltCL(
    Y=trajectories['X'].detach().reshape(nstep + 1, 6), 
    U=trajectories['U'].detach().reshape(nstep, 3), 
    figname=f'cl.png'
)

# %% [markdown]
# ## Save

# %%
if train is True:
    policy_state_dict = {}
    for key, value in best_model.items():
        if "callable." in key:
            new_key = key.split("nodes.0.nodes.1.")[-1]
            policy_state_dict[new_key] = value
    torch.save(policy_state_dict, save_path + f"policy_overtrain.pth")
else:
    print(f"not saving policy as train is: {train}")

# %%

# Extract x and y
x_positions = trajectories['X'][0, :, 0]
y_positions = trajectories['X'][0, :, 2]

# Compute the Euclidean distance for each time step
cyl_distances = torch.sqrt((x_positions - 1)**2 + (y_positions - 1)**2) - radius

# Get the minimum distance and its index (which represents the timestep)
min_cyl_distance, min_index = torch.min(cyl_distances, dim=0)

# Extract the final values of x, y, and z
x_final = trajectories['X'][0, -1, 0]
y_final = trajectories['X'][0, -1, 2]
z_final = trajectories['X'][0, -1, 4]

# Compute the Euclidean distance to (2,2,2)
terminal_distance = torch.sqrt((x_final - 2)**2 + (y_final - 2)**2 + (z_final - 2)**2)

performance_criteria = {
    'minimum_cylinder_distance': min_cyl_distance,
    'terminal_distance': terminal_distance
}

torch.save(performance_criteria, save_path + f"pc_qcon_{Q_con}_qt_{Q_terminal}.pth")

plot_traj(trajectories)

# Evaluation
print('fin')

