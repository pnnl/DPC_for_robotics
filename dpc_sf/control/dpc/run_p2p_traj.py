"""
This script aims to combine the low level PI controller
with the extremely simple DPC trained on a 3 double integrator
setup
"""

# we probably want to learn a gain to apply to the DPC as it
# was trained assuming a certain A and B matrix

import torch
import numpy as np

from neuromancer.modules import blocks
from neuromancer.system import Node, System

from dpc_sf.control.pi.pi import XYZ_Vel
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.dynamics.eom_dpc import QuadcopterDPC
from dpc_sf.control.trajectory.trajectory import waypoint_reference
from dpc_sf.utils import pytorch_utils as ptu
from dpc_sf.utils.animation import Animator

from dpc_sf.control.dpc.operations import processP2PTrajPolicyInput

Ts = 0.001
include_actuators = True
set_vel_zero = False # dont set this true
save_path = "data/policy/DPC_traj/wp_traj.pth"
nstep = 20000
average_vel = 0.5

class stateRefCat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, r):
        e = r - x
        return e # torch.hstack([e,p])

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
    
state_selector = stateSelector()
state_selector_node = Node(state_selector, ['X', 'R'], ['X_reduced', 'R_reduced'], name='state_selector')

state_ref_cat = processP2PTrajPolicyInput(use_error=True)
state_ref_cat_node = Node(state_ref_cat, ['X_reduced', 'R_reduced'], ['Obs'])

mlp = blocks.MLP(6, 3, bias=True,
                 linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ReLU,
                 hsizes=[20, 20, 20, 20])
mlp_node = Node(mlp, ['Obs'], ['xyz_thr'], name='mlp')

mlp_gain = mlpGain()
mlp_gain_node = Node(mlp_gain, ['xyz_thr'], ['xyz_thr_gained'], name='gain')

pi = XYZ_Vel(Ts=Ts, bs=1, input='xyz_thr', include_actuators=include_actuators)
pi_node = Node(pi, ['X', 'xyz_thr_gained'], ['U'], name='pi')

sys = QuadcopterDPC(
    params=quad_params,
    nx=13,
    nu=4,
    ts=Ts,
    normalize=False,
    mean=None,
    var=None,
    include_actuators=include_actuators
)
sys_node = Node(sys, input_keys=['X', 'U'], output_keys=['X'], name='dynamics')

cl_system = System([state_selector_node, state_ref_cat_node, mlp_node, mlp_gain_node, pi_node, sys_node], nsteps=100)

# the reference about which we generate data
R = waypoint_reference('wp_traj', average_vel=1.0, include_actuators=include_actuators)
# r = quad_params["default_init_state_np"]

# test closed loop
print(f"testing closed loop control...")
# test call of closed loop system

# Generate time points using linspace.
time_points = torch.linspace(0, Ts * nstep, nstep)

# Get the references for each time point.
ref_sequence = [ptu.from_numpy(R(t.item())[None,:][None,:].astype(np.float32)) for t in time_points]

# Stack the references on top of each other.
ref_tensor = torch.cat(ref_sequence, axis=1)

P = torch.concat([torch.tensor([[[4,4,4,-5]]])] * (nstep + 1), dim=1)

if include_actuators:
    data = {
        'X': ptu.from_numpy(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32)),
        'R': ref_tensor,
    }
else:
    data = {
        'X': ptu.from_numpy(quad_params["default_init_state_np"][:13][None,:][None,:].astype(np.float32)),
        'R': ref_tensor,
    }

# load the data for the policy
mlp_state_dict = torch.load(save_path)
cl_system.nodes[2].load_state_dict(mlp_state_dict)

# cl_system.nodes[0].callable.vel_sp_2_w_cmd.bs = data['X'].shape[1]
# cl_system.nodes[0].callable.vel_sp_2_w_cmd.reset() 
cl_system.nsteps = nstep
output = cl_system(data)
print("done")

import matplotlib.pyplot as plt
# plt.plot(data['X'][0,:,0:3].detach().cpu().numpy(), label=['x','y','z'])
# plt.plot(data['R'][0,:,0:3].detach().cpu().numpy(), label=['x_ref','y_ref','z_ref'])
# plt.legend()
# plt.show()

t = np.linspace(0, nstep*Ts, nstep)
render_interval = 60
animator = Animator(
    states=ptu.to_numpy(output['X'].squeeze())[::render_interval,:], 
    times=t[::render_interval], 
    reference_history=ptu.to_numpy(output['R'].squeeze())[::render_interval,:], 
    reference=R, 
    reference_type='fig8', 
    drawCylinder=False,
    state_prediction=None
)
animator.animate() # does not contain plt.show()    
plt.show()

print('fin')
# reset batch expectations of low level control
# l_system.nodes[0].callable.vel_sp_2_w_cmd.bs = num_train_samples
# l_system.nodes[0].callable.vel_sp_2_w_cmd.reset() 