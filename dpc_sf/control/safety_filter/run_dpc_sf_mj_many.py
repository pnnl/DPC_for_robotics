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
from tqdm import tqdm

from neuromancer.modules import blocks
from neuromancer.system import Node, System

from dpc_sf.control.pi.pi import XYZ_Vel
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.dynamics.eom_dpc import QuadcopterDPC
from dpc_sf.dynamics.eom_ca import QuadcopterCA
from dpc_sf.control.trajectory.trajectory import waypoint_reference
from dpc_sf.utils import pytorch_utils as ptu
from dpc_sf.utils.animation import Animator

import casadi as ca
import dpc_sf.control.safety_filter.safetyfilternew as sf

## Options
Ts = 0.001
save_path = "data/policy/DPC_p2p/"
policy_name = "policy_experimental_tt.pth"
normalize = False
include_actuators = True 
backend = 'mj' # 'eom'
use_backup = False
nstep = 5000 # 600 * 5
use_integrator_policy = False
num_trajectories = 8

mlp_state_dict = torch.load(save_path + policy_name)

## Neuromancer System Definition
### Classes:
class Cat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, r, c):
        # expects shape [bs, nx]
        return torch.hstack([r-x,c])

class stateSelector(torch.nn.Module):
    def __init__(self, idx=[0,7,1,8,2,9]) -> None:
        super().__init__()
        self.idx = idx
        self.dist2cyl = 100

    def forward(self, x, r):
        """literally just select x,xd,y,yd,z,zd from state"""
        x_reduced = x[:, self.idx]
        r_reduced = r[:, self.idx]
        # print(f"selected_states: {x_reduced}")
        # print(f"selected_states: {r_reduced}")
        # print(f"dist2cyl: {}")
        print(f'x = {str(x[0,0])}')
        print(f'y = {str(x[0,1])}')
        print(f'z = {str(x[0,2])}')
        print(f'x_dot = {str(x[0,0+7])}')
        print(f'y_dot = {str(x[0,1+7])}')
        print(f'z_dot = {str(x[0,2+7])}')
        dist2cyl = np.sqrt((x[0,0]-1)**2+(x[0,1]-1)**2) - 0.5
        print(f'dist2cyl = {str(dist2cyl)}')
        self.dist2cyl = min(dist2cyl, self.dist2cyl)
        print(f'min dist2cyl: {self.dist2cyl}')
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
        self.gravity_offset = ptu.tensor([0,0,-quad_params["hover_thr"]]).unsqueeze(0)
    def forward(self, u):
        """literally apply gain to the output"""
        output = u * self.gain + self.gravity_offset
        print(f"gained u: {output}")
        return output
    
class Policy:

    def __init__(self, mlp_state_dict) -> None:

        # CasADI Setup Policy
        # -------------------
        self.mlp = blocks.MLP(6 + 2, 3, bias=True,
                linear_map=torch.nn.Linear,
                nonlin=torch.nn.ReLU,
                hsizes=[20, 20, 20, 20])
        filtered_state_dict = {k.replace('callable.', '', 1): v for k, v in mlp_state_dict.items()}
        self.mlp.load_state_dict(filtered_state_dict)

        self.pi = XYZ_Vel(Ts=Ts, bs=1, input='xyz_thr', include_actuators=include_actuators)

        self.gravity_offset = ptu.tensor([0,0,-quad_params["hover_thr"]]).unsqueeze(0)

    def evaluate(self, xrc_reduced, x):
        xyz_thr_sp = self.mlp(xrc_reduced) + self.gravity_offset
        u = self.pi(x=x, sp=xyz_thr_sp)
        return u
    
    
class safetyFilterNM(torch.nn.Module):
    def __init__(self, mlp_state_dict) -> None:
        super().__init__()

        # instantiate CasADI policy using the pretrained neuromancer weights as the nominal control
        # inputs: 
        #       - xrc_reduced
        #       - x (full state)
        # outputs:
        #       - u (rotor angular acceleration)
        self.policy = Policy(mlp_state_dict=mlp_state_dict)

        # Safety filter Setup
        # -------------------

        # terminal constraint set
        self.xr = np.copy(quad_params["default_init_state_np"])
        self.xr[0] = 2 # x  
        self.xr[1] = 2 # y
        self.xr[2] = 1 # z flipped

        # instantiate the dynamics
        # ------------------------
        quad = QuadcopterCA(params=quad_params)
        dt = Ts
        nx = 17
        def f(x, t, u):
            return quad.state_dot_1d(x, u)[:,0]
        # f = lambda x, t, u: quad.state_dot_1d(x, u)[:,0]

        # disturbance on the input
        # ------------------------
        disturbance_scaling = 0.0
        # idx = [7,8,9,10,11,12] # select velocities to perturb
        idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] # select all states to perturb
        # idx = [14,15,16,17] # perturb just actuators
        # idx = [0,1,2] # perturb position
        # idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13] # perturb all but actuators
        pert_idx = [1 if i in idx else 0 for i in range(nx)]
        pert_mat = np.diag(pert_idx)
        d = lambda x, t, u: pert_mat * np.sin(t) * disturbance_scaling
        f_pert = lambda x, t, u: f(x, t, u) + d(x, t, u)
        N = 200 # 3
        Nsim = 300

        # Safety Filter Parameters
        # ------------------------
        self.params = {}
        self.params['N'] = N  # prediction horizon
        self.params['dt'] = dt  # sampling time
        self.params['delta'] = 0.00005  # Small robustness margin for constraint tightening
        # self.params['delta'] = 0.001
        self.params['robust_margin'] = 0.1 #0.0 #  #   Robustness margin for handling perturbations
        self.params['robust_margin_terminal'] = 0.0001 #0.0 # # Robustness margin for handling perturbations (terminal set)
        self.params['alpha2'] = 1.0  # Multiplication factor for feasibility parameters
        self.params['integrator_type'] = 'Euler'  # Integrator type for the safety filter

        # Define System Constraints
        # -------------------------
        terminal_constraint = {}
        state_constraints = {}
        input_constraints = {}

        self.params['nx'] = 17
        self.params['nu'] = 4

        umax = [quad.params["maxCmd"]]  #model.umax*np.ones((nu,1)) # Input constraint parameters(max)
        umin = [quad.params["minCmd"]] #model.umin*np.ones((nu,1)) # Input constraint parameters (min)

        scu_k = quad.params["state_ub"] 
        scl_k = quad.params["state_lb"]
        scu_k[0:13] *= 100
        scl_k[0:13] *= 100
        
        state_constraints[0] = lambda x, k: x - scu_k
        state_constraints[1] = lambda x, k: scl_k - x
        # cylinder constraint
        x_c = 1.0
        y_c = 1.0
        r = 0.51
        state_constraints[2] = lambda x, k: r**2 * (1 + k * dt * 0.001) - (x[0] - x_c)**2 - (x[1] - y_c)**2
        # state_constraints[2] = lambda x, k: r**2 - (x[0] - x_c)**2 - (x[1] - y_c)**2

        # Quadratic CBF
        cbf_const = (np.sqrt(2) - 0.5) ** 2
        hf = lambda x: (x-self.xr).T @ (x-self.xr) - cbf_const
        # hf = lambda x: r**2 - (x[0] - x_c)**2 - (x[1] - y_c)**2

        # Input constraints: h(u) <= 0
        #input_constraints[0] = lambda u: u - umax
        #input_constraints[1] = lambda u: umin - u

        kk = 0
        nu = 4
        for ii in range(nu):
            input_constraints[kk] = lambda u: u[ii] - umax[0]
            kk += 1
            input_constraints[kk] = lambda u: umin[0] - u[ii]
            kk += 1    

        self.constraints = {}
        self.constraints['state_constraints'] = state_constraints
        self.constraints['hf'] = hf
        self.constraints['input_constraints'] = input_constraints    

        # Implementation
        # --------------
        # Set up the safety filter and nominal control
        # x0 = random_state()[:,None] # 0.9*model.x0.reshape((nx, 1)) #
        x0 = quad_params["default_init_state_np"]
        self.SF = sf.SafetyFilter(x0=x0, f=f, params=self.params, constraints=self.constraints, options={})
        self.CL = sf.ClosedLoopSystem(f=f_pert, u=self.SF, u_nom=self.unom, dt=dt, int_type='Euler', use_true_unom=True)
        # [X, U, x_pred_history, u_pred_history] = CL.simulate(x0=x0, N=Nsim)

        # test open loop rollout
#         Xopl, Uopl, Xi_sol, XiN_sol = SF.open_loop_rollout(unom=self.unom, x0=x0, k0=0)
# 
#         # lets get the animation in here
#         t = np.linspace(0, Nsim*dt, Nsim)
#         r = np.zeros(t.shape)
#         R = waypoint_reference('wp_p2p', average_vel=1.0, include_actuators=True)
# 
#         x_preds = np.stack([np.vstack(lst) for lst in x_pred_history], axis=1).transpose([1,2,0])
#         render_interval = 2
#         animator = Animator(
#             states=X[::render_interval,:], 
#             times=t[::render_interval], 
#             reference_history=X[::render_interval], 
#             reference=R, 
#             reference_type='wp_p2p', 
#             drawCylinder=True,
#             drawTermSet=True,
#             termSetRad=np.sqrt(cbf_const),
#             state_prediction=x_preds[::render_interval,:,:]
#         )
#         animator.animate() # does not contain plt.show()    
#         plt.show()

    def forward(self, x):
        
        u, x_pred, u_pred = self.SF.compute_control_step(self.unom, ptu.to_numpy(x.flatten()), k=0)

        return ptu.from_numpy(u.T)


    def get_xrc_reduced(self, x):

        state_idx = [0,7,1,8,2,9]
        x_reduced = x[state_idx]
        r_reduced = self.xr[state_idx]
        cyl = np.array([1,1])
        return np.hstack([r_reduced - x_reduced, cyl])

    def unom(self, x, k):
        if type(x) is ca.DM:
            x = np.array(x.full())[:,0]
        xrc_reduced = self.get_xrc_reduced(x)
        switch_z = False
        if switch_z is True:
            xrc_reduced[4:6] *= -1
        # switch to pytorch
        xrc_reduced_pt = ptu.from_numpy(xrc_reduced).unsqueeze(0)
        x_pt = ptu.from_numpy(x).unsqueeze(0)
        u_pt = self.policy.evaluate(xrc_reduced_pt, x_pt).squeeze()
        return ptu.to_numpy(u_pt)[:,None]


### Nodes:
node_list = []

state_selector = stateSelector()
state_selector_node = Node(state_selector, ['X', 'R'], ['X_reduced', 'R_reduced'], name='state_selector')
node_list.append(state_selector_node)

state_ref_cat = Cat()
state_ref_cat_node = Node(state_ref_cat, ['X_reduced', 'R_reduced', 'Cyl'], ['XRC_reduced'], name='cat')
node_list.append(state_ref_cat_node)

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

filter = safetyFilterNM(mlp_state_dict=mlp_state_dict)
filter_node = Node(filter, ['X'], ['U_filtered'])
node_list.append(filter_node)

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
sys_node = Node(sys, input_keys=['X', 'U_filtered'], output_keys=['X'], name='dynamics')
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
            'I_err': ptu.create_zeros([1,1,3])
        })

else:
    data = {
        'X': ptu.from_numpy(quad_params["default_init_state_np"][:13][None,:][None,:].astype(np.float32)),
        'R': torch.concatenate([ptu.from_numpy(R(1)[None,:][None,:].astype(np.float32))]*nstep, axis=1),
        'Cyl': torch.concatenate([ptu.tensor([[[1,1]]])]*nstep, axis=1),
        'I_err': ptu.create_zeros([1,1,3])
    }

## Load Pretrained Policy
# load the data for the policy
cl_system.nodes[2].load_state_dict(mlp_state_dict)

print(f"generating new data, num_trajectories: {num_trajectories}")
# load pretrained policy
mlp_state_dict = torch.load(save_path + policy_name)
for idx, dataset in tqdm(enumerate(data)):
    # apply pretrained policy
    cl_system.nodes[2].load_state_dict(mlp_state_dict)
    # reset simulation to correct initial conditions
    cl_system.nodes[6].callable.mj_reset(dataset['X'].squeeze().squeeze())
    # Perform CLP Simulation
    output = cl_system(dataset)
    # save
    np.savez("data/media/paper/sf_data/" + str(idx), output['X'])

## Perform CLP Simulation
cl_system.nsteps = nstep
data = cl_system(data)
print("done")

plt.plot(data['X'][0,:,0:3].detach().cpu().numpy(), label=['x','y','z'])
plt.plot(data['R'][0,:,0:3].detach().cpu().numpy(), label=['x_ref','y_ref','z_ref'])
plt.legend()
# plt.show()

t = np.linspace(0, nstep*Ts, nstep)
render_interval = 30
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

