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
from dpc_sf.dynamics.eom_ca import QuadcopterCA
from dpc_sf.dynamics.eom_pt import QuadcopterPT
from dpc_sf.control.trajectory.trajectory import waypoint_reference
from dpc_sf.utils import pytorch_utils as ptu
from dpc_sf.utils.animation import Animator

import casadi as ca
import dpc_sf.control.safety_filter.safetyfilteroptimised as sf

## Options
Ts = 0.001
save_path = "data/policy/DPC_p2p/" # redundant/"
policy_name = "policy.pth" # _backup2.pth"
normalize = False
include_actuators = True 
backend = 'mj' # 'eom'
use_backup = False
nstep = 3000 # 600 * 5
use_integrator_policy = False
N_pred = 100
Tf_hzn = 3.0

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
        # print(f'x = {str(x[0,0])}')
        # print(f'y = {str(x[0,1])}')
        # print(f'z = {str(x[0,2])}')
        # print(f'x_dot = {str(x[0,0+7])}')
        # print(f'y_dot = {str(x[0,1+7])}')
        # print(f'z_dot = {str(x[0,2+7])}')
        # dist2cyl = np.sqrt((x[0,0]-1).detach().numpy()**2+(x[0,1].detach().numpy()-1)**2) - 0.5
        # print(f'dist2cyl = {str(dist2cyl)}')
        # self.dist2cyl = min(dist2cyl, self.dist2cyl)
        # print(f'min dist2cyl: {self.dist2cyl}')
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
        # print(f"gained u: {output}")
        return output
    
class safetyFilterNM(torch.nn.Module):
    def __init__(self, N_pred, Tf_hzn) -> None:
        super().__init__()

        # instantiate CasADI policy using the pretrained neuromancer weights as the nominal control
        # inputs: 
        #       - xrc_reduced
        #       - x (full state)
        # outputs:
        #       - u (rotor angular acceleration)

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

        # Safety Filter Parameters
        # ------------------------
        self.params = {}
        self.params['N'] = N_pred  # prediction horizon
        self.params['dt'] = dt  # sampling time

            # Find the optimal dts for the MPC
        dt_1 = Ts
        d = (2 * (Tf_hzn/N_pred) - 2 * dt_1) / (N_pred - 1)
        dts_init = [dt_1 + i * d for i in range(N_pred)]

        self.params['dts'] = dts_init
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
        # [X, U, x_pred_history, u_pred_history] = CL.simulate(x0=x0, N=Nsim)

        # Create Closed Loop System to check feasibility
        # --------------------------------------------
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
            backend='eom',
            reference=R
        )
        sys_node = Node(sys, input_keys=['X', 'U'], output_keys=['X'], name='dynamics')
        node_list.append(sys_node)

        # [state_selector_node, state_ref_cat_node, mlp_node, mlp_gain_node, pi_node, sys_node]
        self.unfiltered_cl_system = System(node_list, nsteps=100)

        self.unfiltered_cl_system.nodes[2].load_state_dict(mlp_state_dict)

        # For state display:
        self.dist2cyl = 100

    def forward(self, x, unom):

        self.display_state(x)

        feas, x_pred, u_pred = self.check_dpc_feas(x0=x)
        x_pred.detach(), u_pred.detach()

        # feas = False
        
        if feas is True:
            self.SF.last_feas_u = ptu.to_numpy(unom)
            return unom #, None, None
        else:
            u = self.SF(ptu.to_numpy(x.flatten()), ptu.to_numpy(unom.flatten()))
            return ptu.from_numpy(u) #,  x_pred, u_pred
        
    def display_state(self, x):

        # idx = [0,7,1,8,2,9]
        # x_reduced = x[:, idx]
        # r_reduced = r[:, self.idx]
        # print(f"selected_states: {x_reduced}")
        # print(f"selected_states: {r_reduced}")
        # print(f"dist2cyl: {}")
        print(f'x = {str(x[0,0])}')
        print(f'y = {str(x[0,1])}')
        # print(f'z = {str(x[0,2])}')
        # print(f'x_dot = {str(x[0,0+7])}')
        # print(f'y_dot = {str(x[0,1+7])}')
        # print(f'z_dot = {str(x[0,2+7])}')
        dist2cyl = np.sqrt((x[0,0]-1)**2+(x[0,1]-1)**2) - 0.5
        # print(f'dist2cyl = {str(dist2cyl)}')
        self.dist2cyl = min(dist2cyl, self.dist2cyl)
        print(f'min dist2cyl: {self.dist2cyl}')
    
    def check_dpc_feas(self, x0):

        # check that the states of the rollout of the pytorch quad
        # from x0 under unom remain within bounds for N steps
        print("checking unfiltered feasibility")
        if type(x0) is np.ndarray:
            #self.unfiltered_cl_system.nodes[5].callable.reset(x0)
            x = ptu.from_numpy(x0)
        elif type(x0) is torch.Tensor:
            # self.unfiltered_cl_system.nodes[5].callable.reset(ptu.to_numpy(x0))
            x = torch.clone(x0)
        elif type(x0) is ca.DM:
            #self.unfiltered_cl_system.nodes[5].callable.reset(x0.full()[:,0])
            x = ptu.from_numpy(np.copy(x0.full()[:,0]))
        else:
            raise Exception(f"unexpected input type: {type(x0)}")
        
        data = {
            'X': x.unsqueeze(0), # ptu.from_numpy(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32)),
            'R': torch.concatenate([ptu.from_numpy(R(1)[None,:][None,:].astype(np.float32))]*self.unfiltered_cl_system.nsteps, axis=1),
            'Cyl': torch.concatenate([ptu.tensor([[[1,1]]])]*self.unfiltered_cl_system.nsteps, axis=1),
        }

        # simulate
        output = self.unfiltered_cl_system(data)

        lb = ptu.from_numpy(quad_params["state_lb"])
        ub = ptu.from_numpy(quad_params["state_ub"])

        for idx, x in enumerate(output['X'][0]):
            if not torch.all(x[:13] >= lb[:13]) or not torch.all(x[:13] <= ub[:13]):
                print("VIOLATION: state constraint predicted to be violated")
                print(f"violation occured {idx} timesteps in the future")
                return False, output['X'][0], output['U'][0]
            # Cylinder constraint check
            r = 0.51
            x_c = 1.0
            y_c = 1.0
            if r**2 - (x[0] - x_c)**2 - (x[1] - y_c)**2 > 0:  # If this is positive, constraint is violated
                print("VIOLATION: cylinder constraint predicted to be violated")
                print(f"violation occured {idx} timesteps in the future")
                return False, output['X'][0], output['U'][0]
            
        print("NOMINAL: no constraints predicted to be violated")
        return True, output['X'][0], output['U'][0]


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

filter = safetyFilterNM(N_pred = N_pred, Tf_hzn = Tf_hzn)
filter_node = Node(filter, ['X', 'U'], ['U_filtered'])
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
        'Cyl': torch.concatenate([ptu.tensor([[[1,1]]])]*nstep, axis=1),
    }
else:
    data = {
        'X': ptu.from_numpy(quad_params["default_init_state_np"][:13][None,:][None,:].astype(np.float32)),
        'R': torch.concatenate([ptu.from_numpy(R(1)[None,:][None,:].astype(np.float32))]*nstep, axis=1),
        'Cyl': torch.concatenate([ptu.tensor([[[1,1]]])]*nstep, axis=1),
    }

## Load Pretrained Policy
# load the data for the policy
cl_system.nodes[2].load_state_dict(mlp_state_dict)

## Perform CLP Simulation
cl_system.nsteps = nstep
data = cl_system(data)
print("done")

plt.plot(data['X'][0,:,0:3].detach().cpu().numpy(), label=['x','y','z'])
plt.plot(data['R'][0,:,0:3].detach().cpu().numpy(), label=['x_ref','y_ref','z_ref'])
plt.legend()
plt.show()

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


