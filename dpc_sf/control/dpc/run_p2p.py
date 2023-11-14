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

## Neuromancer System Definition
### Classes:
class stateSelector(torch.nn.Module):
    def __init__(self, idx=[0,7,1,8,2,9], radius=0.5) -> None:
        super().__init__()
        self.idx = idx
        self.radius=radius

    def forward(self, x, r, cyl=None):
        """literally just select x,xd,y,yd,z,zd from state"""
        x_reduced = x[:, self.idx]
        r_reduced = r[:, self.idx]
        # print(f"selected_states: {x_reduced}")
        # print(f"selected_states: {r_reduced}")
        # clip selected states to be fed into the agent:
        x_reduced = torch.clip(x_reduced, min=ptu.tensor(-3.), max=ptu.tensor(3.))
        r_reduced = torch.clip(r_reduced, min=ptu.tensor(-3.), max=ptu.tensor(3.))
        # generate the errors
        e = r_reduced - x_reduced
        c_pos, c_vel = posVel2cyl(x_reduced, cyl, self.radius)
        return torch.hstack([e, c_pos, c_vel])

class mlpGain(torch.nn.Module):
    def __init__(
            self, 
            gain= ptu.tensor([1.0,1.0,1.0])#torch.tensor([-0.1, -0.1, -0.01])
        ) -> None:
        super().__init__()
        self.gain = gain # * 0.1
        self.gravity_offset = ptu.tensor([0,0,-quad_params["hover_thr"]]).unsqueeze(0)
    def forward(self, u):
        """literally apply gain to the output"""
        output = u * self.gain + self.gravity_offset
        # print(f"gained u: {output}")
        return output
    
def run_dpc_p2p(
        Ts                = 0.001,
        save_path         = "data/policy/DPC_p2p/",
        policy_name       = "policy.pth",
        normalize         = False,
        include_actuators = True,
        backend           = 'mj',
        nstep             = 5000,
        radius            = 0.5,
    ):


    node_list = []

    state_selector = stateSelector()
    state_selector_node = Node(state_selector, ['X', 'R', 'Cyl'], ['XRC_reduced'], name='state_selector')
    node_list.append(state_selector_node)

    mlp = blocks.MLP(6 + 2, 3, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU,
                    hsizes=[20, 20, 20, 20]).to(ptu.device)
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
        # X = torch.tensor([[2, 2]], dtype=torch.float32)  # Assuming a sample initial state
        X = ptu.from_numpy(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32))
        data = {
            'X': X,
            'R': torch.concatenate([ptu.from_numpy(R(1)[None,:][None,:].astype(np.float32))]*nstep, axis=1),
            'Cyl': torch.concatenate([ptu.tensor([[[1,1]]])]*nstep, axis=1),
        }

    else:
        data = {
            'X': ptu.from_numpy(quad_params["default_init_state_np"][:13][None,:][None,:].astype(np.float32)),
            'R': torch.concatenate([ptu.from_numpy(R(1)[None,:][None,:].astype(np.float32))]*nstep, axis=1),
            'Cyl': torch.concatenate([ptu.tensor([[[1,1]]])]*nstep, axis=1),
        }

    print(f"generating new data")
    # load pretrained policy
    mlp_state_dict = torch.load(save_path + policy_name)
    # apply pretrained policy
    cl_system.nodes[1].load_state_dict(mlp_state_dict)
    # reset simulation to correct initial conditions
    cl_system.nodes[4].callable.mj_reset(ptu.to_numpy(data['X'].squeeze().squeeze()))
    # Perform CLP Simulation
    data = cl_system(data)
    # save
    # np.savez(save_path + str(idx), data['X'])
    print("saving the state and input histories...")
    x_history = np.stack(ptu.to_numpy(data['X'].squeeze()))
    u_history = np.stack(ptu.to_numpy(data['U'].squeeze()))

    np.savez(
        file = f"data/dpc_timehistories/xu_p2p_mj_{str(Ts)}.npz",
        x_history = x_history,
        u_history = u_history
    )

    


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    run_dpc_p2p()