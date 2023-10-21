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
from dpc_sf.control.pi.pi_ca import XYZ_Vel as XYZ_Vel_CA
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.dynamics.eom_dpc import QuadcopterDPC
from dpc_sf.dynamics.eom_ca import QuadcopterCA
from dpc_sf.control.trajectory.trajectory import waypoint_reference
from dpc_sf.utils import pytorch_utils as ptu
from dpc_sf.utils.animation import Animator

import casadi as ca
import ml_casadi.torch as mc
import dpc_sf.control.safety_filter.safetyfilter as sf

## Options
Ts = 0.001
save_path = "data/policy/DPC_p2p/"
policy_name = "policy.pth"
normalize = False
include_actuators = True 
backend = 'mj' # 'eom'
use_backup = False
nstep = 10000
use_integrator_policy = False

mlp_state_dict = torch.load(save_path + policy_name)

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
    
class casadiPolicy:

    def __init__(self, mlp_state_dict) -> None:

        # CasADI Setup Policy
        # -------------------
        self.mlp = blocks.MLP(6*2 + 2, 3, bias=True,
                linear_map=torch.nn.Linear,
                nonlin=torch.nn.ReLU,
                hsizes=[20, 20, 20, 20])
        filtered_state_dict = {k.replace('callable.', '', 1): v for k, v in mlp_state_dict.items()}
        self.mlp.load_state_dict(filtered_state_dict)

        model = mc.nn.MultiLayerPerceptron(
            input_size=14, 
            hidden_size=20, 
            output_size=3, 
            n_hidden=4, 
        activation='Tanh')

        # Define a key mapping
        key_mapping = {
            'linear.0.weight': 'input_layer.weight', 
            'linear.0.bias': 'input_layer.bias',
            'linear.1.weight': 'hidden_layers.0.weight', 
            'linear.1.bias': 'hidden_layers.0.bias',
            'linear.2.weight': 'hidden_layers.1.weight', 
            'linear.2.bias': 'hidden_layers.1.bias',
            'linear.3.weight': 'hidden_layers.2.weight', 
            'linear.3.bias': 'hidden_layers.2.bias',
            'linear.4.weight': 'output_layer.weight', 
            'linear.4.bias': 'output_layer.bias'
        }

        # Map the filtered_state_dict keys to model2's state_dict keys
        matched_state_dict = {key_mapping[key]: value for key, value in filtered_state_dict.items()}
        model.load_state_dict(matched_state_dict)

        ## Export the model as Casadi Function
        xrc_reduced = ca.MX.sym('xrc_reduced', 14)
        thr_xyz_sp = model(xrc_reduced)
        self.mlp_casadi_func = ca.Function('model2',
                                [xrc_reduced],
                                [thr_xyz_sp])

        self.pi = XYZ_Vel_CA(Ts=Ts, bs=1, input='xyz_thr', include_actuators=include_actuators)

        x = ca.MX.sym("x_test", 17)

        u = self.pi(x=x, sp=thr_xyz_sp)

        # Store the system as a CasADi function
        self.system_function = ca.Function('system_function', [xrc_reduced, x], [u])

    def evaluate(self, xrc_reduced, x):
        return self.system_function(xrc_reduced, x)
    
    
class safetyFilterNM(torch.nn.Module):
    def __init__(self, mlp_state_dict) -> None:
        super().__init__()

        # instantiate CasADI policy using the pretrained neuromancer weights as the nominal control
        # inputs: 
        #       - xrc_reduced
        #       - x (full state)
        # outputs:
        #       - u (rotor angular acceleration)
        self.casadi_policy = casadiPolicy(mlp_state_dict=mlp_state_dict)

        # Safety filter Setup
        # -------------------

        # terminal constraint set
        self.xr = np.copy(quad_params["default_init_state_np"])
        self.xr[0] = 2 # x  
        self.xr[1] = 2 # y
        self.xr[2] = 1 # z

        # instantiate the dynamics
        # ------------------------
        quad = QuadcopterCA()
        dt = 0.001
        nx = 17
        def f(x, t, u):
            return quad.state_dot_1d(x, u)[:,0]
        # f = lambda x, t, u: quad.state_dot_1d(x, u)[:,0]

        # disturbance on the input
        # ------------------------
        disturbance_scaling = 0.001
        # idx = [7,8,9,10,11,12] # select velocities to perturb
        idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] # select all states to perturb
        # idx = [14,15,16,17] # perturb just actuators
        # idx = [0,1,2] # perturb position
        # idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13] # perturb all but actuators
        pert_idx = [1 if i in idx else 0 for i in range(nx)]
        pert_mat = np.diag(pert_idx)
        d = lambda x, t, u: pert_mat * np.sin(t) * disturbance_scaling
        f_pert = lambda x, t, u: f(x, t, u) + d(x, t, u)
        N = 30
        Nsim = 50

        # Safety Filter Parameters
        # ------------------------
        self.params = {}
        self.params['N'] = N  # prediction horizon
        self.params['dt'] = dt  # sampling time
        #params['delta'] = 0.00005  # Small robustness margin for constraint tightening
        self.params['delta'] = 0.001
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

        state_constraints[0] = lambda x, k: x - scu_k
        state_constraints[1] = lambda x, k: scl_k - x
        # state_constraints[2] = lambda x, k: 

        # Quadratic CBF
        hf = lambda x: (x-self.xr).T @ (x-self.xr) - 0.1

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
        SF = sf.SafetyFilter(x0=x0, f=f, params=self.params, constraints=self.constraints, options={})
        CL = sf.ClosedLoopSystem(f=f_pert, u=SF, u_nom=self.unom, dt=dt, int_type='Euler')
        [X, U] = CL.simulate(x0=x0, N=Nsim)

        # test open loop rollout
        Xopl, Uopl, Xi_sol, XiN_sol = SF.open_loop_rollout(unom=self.unom, x0=x0, k0=0)

        # lets get the animation in here
        t = np.linspace(0, Nsim*dt, Nsim)
        r = np.zeros(t.shape)
        R = waypoint_reference('wp_p2p', average_vel=1.0, include_actuators=True)

        render_interval = 1
        animator = Animator(
            states=X[::render_interval,:], 
            times=t[::render_interval], 
            reference_history=X[::render_interval], 
            reference=R, 
            reference_type='wp_p2p', 
            drawCylinder=True,
            state_prediction=None
        )
        animator.animate() # does not contain plt.show()    
        plt.show()

    def test_sf(self, x0):
        pass

    def get_xrc_reduced(self, x):
        state_idx = [0,7,1,8,2,9]
        x_reduced = x[state_idx]
        r_reduced = self.xr[state_idx]
        cyl = np.array([1,1])
        return ca.vertcat(x_reduced, r_reduced, cyl).T

    def unom(self, x, k):
        xrc_reduced = self.get_xrc_reduced(x)
        u = self.casadi_policy.evaluate(xrc_reduced, x)
        return np.array(u.T.full())

    def forward(self, x):
        pass


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


