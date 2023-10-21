"""
Run Pretrained DPC on Real Quadcopter
-------------------------------------
This script combines the low level PI controller with the extremely
simple DPC trained on a 3 double integrator setup.
"""

## Imports
import torch
import numpy as np
import matplotlib.pyplot as plt

from neuromancer.modules import blocks
from neuromancer.system import Node, System

from dpc_sf.control.pi.pi import XYZ_Vel
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.dynamics.eom_dpc import QuadcopterDPC
from dpc_sf.control.trajectory.trajectory import waypoint_reference
from dpc_sf.utils import pytorch_utils as ptu
from dpc_sf.utils.animation import Animator

## Options
Ts = 0.005
save_path = "data/policy/DPC_p2p/"
policy_name = "policy_overtrain.pth"
normalize = False
include_actuators = True 
backend = 'mj' # 'eom'
use_backup = False
nstep = 2000
use_integrator_policy = False
task = 'fig8'

## Neuromancer System Definition
### Classes:
class Cat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args):
        # expects shape [bs, nx]
        return torch.hstack(args)

class stateSelector(torch.nn.Module):
    def __init__(self, idx=[0,7,1,8,2,9]) -> None:
        super().__init__()
        self.idx = idx

    def forward(self, x, r):
        """literally just select x,xd,y,yd,z,zd from state"""
        x_reduced = x[:, self.idx]
        r_reduced = r[:, self.idx]
        print(f"selected_states: {x_reduced}")
        print(f"selected_states: {r_reduced}")
        # clip selected states to be fed into the agent:
        x_reduced = torch.clip(x_reduced, min=torch.tensor(-3.), max=torch.tensor(3.))
        r_reduced = torch.clip(r_reduced, min=torch.tensor(-3.), max=torch.tensor(3.))
        return x_reduced, r_reduced

class mlpGain(torch.nn.Module):
    def __init__(
            self, 
            gain= torch.tensor([1.0,1.0,1.0])#torch.tensor([-0.1, -0.1, -0.01])
        ) -> None:
        super().__init__()
        self.gain = gain
        self.gravity_offset = ptu.create_tensor([0,0,-quad_params["hover_thr"]]).unsqueeze(0)
    def forward(self, u):
        """literally apply gain to the output"""
        output = u * self.gain + self.gravity_offset
        print(f"gained u: {output}")
        return output

### Nodes:
node_list = []

state_selector = stateSelector()
state_selector_node = Node(state_selector, ['X', 'R'], ['X_reduced', 'R_reduced'], name='state_selector')
node_list.append(state_selector_node)

state_ref_cat = Cat()
state_ref_cat_node = Node(state_ref_cat, ['X_reduced', 'R_reduced', 'Cyl'], ['XRC_reduced'], name='cat')
node_list.append(state_ref_cat_node)

mlp = blocks.MLP(6*2 + 2, 3, bias=True,
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
    data = {
        'X': ptu.from_numpy(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32)),
        'R': torch.concatenate([ptu.from_numpy(R(1)[None,:][None,:].astype(np.float32))]*nstep, axis=1),
        'Cyl': torch.concatenate([ptu.create_tensor([[[1,1]]])]*nstep, axis=1),
        'I_err': ptu.create_zeros([1,1,3])
    }
else:
    data = {
        'X': ptu.from_numpy(quad_params["default_init_state_np"][:13][None,:][None,:].astype(np.float32)),
        'R': torch.concatenate([ptu.from_numpy(R(1)[None,:][None,:].astype(np.float32))]*nstep, axis=1),
        'Cyl': torch.concatenate([ptu.create_tensor([[[1,1]]])]*nstep, axis=1),
        'I_err': ptu.create_zeros([1,1,3])
    }

## Load Pretrained Policy
# load the data for the policy
mlp_state_dict = torch.load(save_path + policy_name)
cl_system.nodes[2].load_state_dict(mlp_state_dict)


## Perform CLP Simulation
cl_system.nsteps = nstep
cl_system(data)
print("done")

plt.plot(data['X'][0,:,0:3].detach().cpu().numpy(), label=['x','y','z'])
plt.plot(data['R'][0,:,0:3].detach().cpu().numpy(), label=['x_ref','y_ref','z_ref'])
plt.legend()
# plt.show()

t = np.linspace(0, nstep*Ts, nstep)
render_interval = 6
animator = Animator(
    states=ptu.to_numpy(data['X'].squeeze())[::render_interval,:], 
    times=t[::render_interval], 
    reference_history=ptu.to_numpy(data['R'].squeeze())[::render_interval,:], 
    reference=R, 
    reference_type='wp_p2p', 
    drawCylinder=True,
    state_prediction=None
)
animator.animate() # does not contain plt.show()    
plt.show()


