"""
Run Pretrained DPC on Real Quadcopter
-------------------------------------
This script combines the low level PI controller with the extremely
simple DPC trained on a 3 double integrator setup.
"""
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse

from neuromancer.modules import blocks
from neuromancer.system import Node, System

from dpc_sf.control.pi.pi import XYZ_Vel
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.dynamics.eom_dpc import QuadcopterDPC
from dpc_sf.control.trajectory.trajectory import waypoint_reference
from dpc_sf.utils import pytorch_utils as ptu
from dpc_sf.utils.animation import Animator

from dpc_sf.control.dpc.operations import posVel2cyl


parser = argparse.ArgumentParser(description="Description of your program")

parser.add_argument("--Ts",                 type=float, default=0.001, help="Your description for Ts")
parser.add_argument("--save_path",          type=str,   default="data/policy/DPC_p2p/", help="Path to save data")
parser.add_argument("--policy_name",        type=str,   default="policy_experimental_tt.pth", help="Name of the policy file")
parser.add_argument("--normalize",          type=bool,  default=False, help="Whether to normalize or not")
parser.add_argument("--include_actuators",  type=bool,  default=True, help="Whether to include actuators or not")
parser.add_argument("--backend",            type=str,   default='mj', choices=['mj', 'eom'], help="Backend to use, 'mj' or 'eom'")
parser.add_argument("--nstep",              type=int,   default=10000, help="Number of steps")
parser.add_argument("--radius",             type=float, default=0.5, help="Radius")
parser.add_argument("--num_trajectories",   type=int,   default=9)
parser.add_argument("--generate_new_data",  type=bool,  default=True)
parser.add_argument("--x_min",              type=float, default=-1.)
parser.add_argument("--x_max",              type=float, default=1.)
parser.add_argument("--y_min",              type=float, default=-1.)
parser.add_argument("--y_max",              type=float, default=1.)
parser.add_argument("--animate_traj_idx",   type=int,   default=4)

args = parser.parse_args()

Ts                  = args.Ts
save_path           = args.save_path
policy_name         = args.policy_name
normalize           = args.normalize
include_actuators   = args.include_actuators 
backend             = args.backend
nstep               = args.nstep
radius              = args.radius
num_trajectories    = args.num_trajectories
generate_new_data   = args.generate_new_data
x_min               = args.x_min
x_max               = args.x_max
y_min               = args.y_min
y_max               = args.y_max
animate_traj_idx    = args.animate_traj_idx

## Neuromancer System Definition
### Classes:
class stateSelector(torch.nn.Module):
    def __init__(self, idx=[0,7,1,8,2,9]) -> None:
        super().__init__()
        self.idx = idx

    def forward(self, x, r, cyl=None):
        """literally just select x,xd,y,yd,z,zd from state"""
        x_reduced = x[:, self.idx]
        r_reduced = r[:, self.idx]
        # print(f"selected_states: {x_reduced}")
        # print(f"selected_states: {r_reduced}")
        # clip selected states to be fed into the agent:
        x_reduced = torch.clip(x_reduced, min=torch.tensor(-3.), max=torch.tensor(3.))
        r_reduced = torch.clip(r_reduced, min=torch.tensor(-3.), max=torch.tensor(3.))
        # generate the errors
        e = r_reduced - x_reduced
        c_pos, c_vel = posVel2cyl(x_reduced, cyl, radius)
        return torch.hstack([e, c_pos, c_vel])

class mlpGain(torch.nn.Module):
    def __init__(
            self, 
            gain= torch.tensor([1.0,1.0,1.0])#torch.tensor([-0.1, -0.1, -0.01])
        ) -> None:
        super().__init__()
        self.gain = gain # * 0.1
        self.gravity_offset = ptu.tensor([0,0,-quad_params["hover_thr"]]).unsqueeze(0)
    def forward(self, u):
        """literally apply gain to the output"""
        output = u * self.gain + self.gravity_offset
        # print(f"gained u: {output}")
        return output

### Nodes:
node_list = []

state_selector = stateSelector()
state_selector_node = Node(state_selector, ['X', 'R', 'Cyl'], ['XRC_reduced'], name='state_selector')
node_list.append(state_selector_node)

mlp = blocks.MLP(6 + 2, 3, bias=True,
                linear_map=torch.nn.Linear,
                nonlin=torch.nn.ReLU,
                hsizes=[20, 20, 20, 20])
policy_node = Node(mlp, ['XRC_reduced'], ['xyz_thr'], name='mlp')
node_list.append(policy_node)

mlp_gain = mlpGain()
mlp_gain_node = Node(mlp_gain, ['xyz_thr'], ['xyz_thr_gained'], name='gravity_offset')
node_list.append(mlp_gain_node)

pi = XYZ_Vel(Ts=Ts, bs=1, input='xyz_thr', include_actuators=include_actuators)
pi_node = Node(pi, ['X', 'xyz_thr_gained'], ['U'], name='pi_control')
node_list.append(pi_node)

# the reference about which we generate data
R = waypoint_reference('wp_p2p', average_vel=1.0, include_actuators=include_actuators)
sys = QuadcopterDPC(
    params=quad_params,
    nx=13,
    nu=4,
    ts=Ts,
    normalize=normalize,
    mean=None,
    var=None,
    include_actuators=include_actuators,
    backend=backend,
    reference=R
)
sys_node = Node(sys, input_keys=['X', 'U'], output_keys=['X'], name='dynamics')
node_list.append(sys_node)

# [state_selector_node, state_ref_cat_node, mlp_node, mlp_gain_node, pi_node, sys_node]
cl_system = System(node_list, nsteps=nstep)

## Dataset Generation
# Here we need only produce one starting point from which to conduct a rollout.
if include_actuators:

    h, k = 2, 2  # center of the circle
    r = np.sqrt(8)  # radius of the circle

    # Assuming theta_min and theta_max have been calculated based on your requirements
    theta_average = torch.pi * 1.25  # the angle for the point (0,0)
    theta_delta = 0.2
    theta_min = theta_average - theta_delta
    theta_max = theta_average + theta_delta
    thetas = np.linspace(theta_min, theta_max, num_trajectories)

    data = []

    # X = torch.tensor([[2, 2]], dtype=torch.float32)  # Assuming a sample initial state
    X = ptu.from_numpy(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32))
    for theta in thetas:
        x = h + r * np.cos(theta)
        y = k + r * np.sin(theta)
        X_iter = X.clone()
        X_iter[..., 0] = x
        X_iter[..., 1] = y
        data.append({
            'X': X_iter,
            'R': torch.concatenate([ptu.from_numpy(R(1)[None,:][None,:].astype(np.float32))]*nstep, axis=1),
            'Cyl': torch.concatenate([ptu.tensor([[[1,1]]])]*nstep, axis=1),
        })

else:
    data = {
        'X': ptu.from_numpy(quad_params["default_init_state_np"][:13][None,:][None,:].astype(np.float32)),
        'R': torch.concatenate([ptu.from_numpy(R(1)[None,:][None,:].astype(np.float32))]*nstep, axis=1),
        'Cyl': torch.concatenate([ptu.tensor([[[1,1]]])]*nstep, axis=1),
    }

if generate_new_data is True:
    print(f"generating new data, num_trajectories: {num_trajectories}")
    # load pretrained policy
    mlp_state_dict = torch.load(save_path + policy_name)
    for idx, dataset in tqdm(enumerate(data)):
        # apply pretrained policy
        cl_system.nodes[1].load_state_dict(mlp_state_dict)
        # reset simulation to correct initial conditions
        cl_system.nodes[4].callable.mj_reset(dataset['X'].squeeze().squeeze())
        # Perform CLP Simulation
        output = cl_system(dataset)
        # save
        np.savez(save_path + str(idx), output['X'])


print("done")

# Plotting
# --------

fig = plt.figure(figsize=(15, 5))

# Function to draw a cylinder
def draw_cylinder(ax, x_center=0, y_center=0, z_center=0, radius=1, depth=1, resolution=100):
    z = np.linspace(z_center - depth, z_center + depth, resolution)
    theta = np.linspace(0, 2*np.pi, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + x_center
    y_grid = radius * np.sin(theta_grid) + y_center
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, rstride=5, cstride=5, color='b')


# Left 3D Plot
ax1 = fig.add_subplot(131, projection='3d')
ax1.set_title('3D Trajectories')
for idx in range(num_trajectories):
    file_path = save_path + str(idx) + ".npz"
    with np.load(file_path) as data:
        trajectory = data['arr_0']
        x = trajectory[0, 1:, 0]
        y = trajectory[0, 1:, 1]
        z = -trajectory[0, 1:, 2]  # Invert z values and ignore the first timestep
        ax1.plot(x, y, z, label=f'Trajectory {idx}')
ax1.scatter(2,2,-1,color='red')

# Set limits for equal axes
max_limit = np.max(np.abs(np.array(ax1.get_xlim() + ax1.get_ylim() + ax1.get_zlim())))
ax1.set_xlim(-1.2, 2.2)
ax1.set_ylim(-1.2, 2.2)
ax1.set_zlim(-2, 2)

# Draw Cylinder manually
draw_cylinder(ax1, x_center=1, y_center=1, z_center=0, radius=0.5, depth=2)  # depth=2 to draw a long cylinder


# Middle Top Down x,y Plot
ax2 = fig.add_subplot(132)
ax2.set_title('Top Down View')
ax2.set_aspect('equal', 'box')
for idx in range(num_trajectories):
    file_path = save_path + str(idx) + ".npz"
    with np.load(file_path) as data:
        trajectory = data['arr_0']
        x = trajectory[0, 1:, 0]
        y = trajectory[0, 1:, 1]
        ax2.plot(x, y, label=f'Trajectory {idx}')

circle = Circle((1, 1), 0.5, fill=False, color='b', linestyle='dashed')
ax2.add_patch(circle)
ax2.scatter(2,2, color='red')

# Right y,z Front View Plot
ax3 = fig.add_subplot(133)
ax3.set_title('Front View')
ax3.set_aspect('equal', 'box')
for idx in range(num_trajectories):
    file_path = save_path + str(idx) + ".npz"
    with np.load(file_path) as data:
        trajectory = data['arr_0']
        y = trajectory[0, 1:, 1]
        z = -trajectory[0, 1:, 2]  # Invert z values and ignore the first timestep
        ax3.plot(y, z, label=f'Trajectory {idx}')
ax3.scatter(2,-1,color='red')

plt.tight_layout()
plt.show()

reference_history = np.copy(quad_params["default_init_state_np"])
reference_history[0] += 2
reference_history[1] += 2
reference_history[2] += 1
reference_history = np.stack([reference_history] * (nstep + 1))
file_path = save_path + str(animate_traj_idx) + ".npz"
with np.load(file_path) as data:
    # print(data)

    # for idx in range(num_trajectories):
    # plt.plot(data['X'][0,:,0:3].detach().cpu().numpy(), label=['x','y','z'])
    # #plt.plot(data['R'][0,:,0:3].detach().cpu().numpy(), label=['x_ref','y_ref','z_ref'])
    # plt.legend()
    # plt.show()
    # 
    t = np.linspace(0, nstep*Ts, nstep)
    render_interval = 60
    animator = Animator(
        states=data["arr_0"][0][::render_interval,:], 
        times=t[::render_interval], 
        reference_history=reference_history[::render_interval,:], 
        reference=R, 
        reference_type='wp_p2p', 
        drawCylinder=True,
        state_prediction=None
    )
    animator.animate() # does not contain plt.show()    
    plt.show()


