import traceback
import numpy as np
import torch
import neuromancer as nm
from neuromancer.dynamics import integrators
import casadi as ca
import copy
import time

import utils.pytorch as ptu
from dpc import posVel2cyl
from utils.integrate import euler, RK4, generate_variable_timesteps, generate_variable_times
from dynamics import get_quad_params, mujoco_quad, state_dot
from pid import get_ctrl_params, PID
import reference
from utils.quad import Animator

from safety_filter_example import SafetyFilter

def run_nav(
        Ti, Tf, Ts,
        N_sf,
        integrator = 'euler',
        policy_save_path = 'data/',
        media_save_path = 'data/training/',
    ):

    times = np.arange(Ti, Tf, Ts)
    nsteps = len(times)
    quad_params = get_quad_params()
    ctrl_params = get_ctrl_params()
    mlp_state_dict = torch.load(policy_save_path + 'wp_p2p_policy.pth')

    # SF = sf.SafetyFilter(x0=x0, f=f, params=params, constraints=constraints, options=options)
    safety_filter = SafetyFilter(state_dot.casadi_vectorized, Ts, Tf_hzn_sf, N_sf, quad_params, euler)
    R = reference.waypoint('wp_p2p', average_vel=1.0)

    node_list = []

    def process_policy_input(x, r, c, radius=0.5):
        idx = [0,7,1,8,2,9]
        x_r, r_r = x[:, idx], r[:, idx]
        x_r = torch.clip(x_r, min=ptu.tensor(-3.), max=ptu.tensor(3.))
        r_r = torch.clip(r_r, min=ptu.tensor(-3.), max=ptu.tensor(3.))
        c_pos, c_vel = posVel2cyl(x_r, c, radius)
        return torch.hstack([r_r - x_r, c_pos, c_vel])
    process_policy_input_node = nm.system.Node(process_policy_input, ['X', 'R', 'Cyl'], ['Obs'], name='state_selector')
    node_list.append(process_policy_input_node)

    mlp = nm.modules.blocks.MLP(6 + 2, 3, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU,
                    hsizes=[20, 20, 20, 20]).to(ptu.device)
    policy_node = nm.system.Node(mlp, ['Obs'], ['MLP_U'], name='mlp')
    node_list.append(policy_node)

    gravity_offset = lambda u: u + ptu.tensor([[0,0,-quad_params["hover_thr"]]])
    gravity_offset_node = nm.system.Node(gravity_offset, ['MLP_U'], ['MLP_U_grav'], name='gravity_offset')
    node_list.append(gravity_offset_node)

    pid = PID(Ts=Ts, bs=1, ctrl_params=ctrl_params, quad_params=quad_params)
    pid_node = nm.system.Node(pid, ['X', 'MLP_U_grav'], ['U', 'PID_X'], name='pid_control')
    node_list.append(pid_node)

    filter = lambda x_seq, u_seq, violation: safety_filter(x_seq, u_seq, violation).unsqueeze(0)
    filter_node = nm.system.Node(filter, input_keys=['X', 'U'], output_keys=['U_filtered'], name='dynamics')
    node_list.append(filter_node)

    sys = mujoco_quad(state=quad_params["default_init_state_np"], quad_params=quad_params, Ti=Ti, Tf=Tf, Ts=Ts, integrator=integrator)
    sys_node = nm.system.Node(sys, input_keys=['X', 'U_filtered'], output_keys=['X'], name='dynamics')
    node_list.append(sys_node)

    cl_system = nm.system.System(node_list, nsteps=nsteps)

    # load the pretrained policies
    cl_system.nodes[1].load_state_dict(mlp_state_dict)

    X = ptu.from_numpy(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32))
    X[:,:,7] = 1.5 # xdot adversarial
    X[:,:,8] = 1.5 # ydot adversarial

    # X[:,:,0] = 0.45
    # X[:,:,1] = 0.45
    data = {
        'X': X,
        'R': torch.concatenate([ptu.from_numpy(R(1)[None,:][None,:].astype(np.float32))]*nsteps, axis=1),
        'Cyl': torch.concatenate([ptu.tensor([[[1,1]]])]*nsteps, axis=1),
    }

    # set the mujoco simulation to the correct initial conditions
    cl_system.nodes[6].callable.set_state(ptu.to_numpy(data['X'].squeeze().squeeze()))

    # Perform CLP Simulation
    output = cl_system.forward(data, retain_grad=False, print_loop=True)

    # save
    print("saving the state and input histories...")
    x_history = np.stack(ptu.to_numpy(output['X'].squeeze()))
    u_history = np.stack(ptu.to_numpy(output['U'].squeeze()))
    r_history = np.stack(ptu.to_numpy(output['R'].squeeze()))

    np.savez(
        file = f"data/xu_sf_p2p_mj_{str(Ts)}.npz",
        x_history = x_history,
        u_history = u_history,
        r_history = r_history
    )

    animator = Animator(x_history, times, r_history, max_frames=500, save_path=media_save_path, state_prediction=None, drawCylinder=True)
    animator.animate()

    print('fin')

def run_nav_no_event_trigger(
        Ti, Tf, Ts,
        N_sf, N_pred, Tf_hzn_sf, Tf_hzn_pred,
        integrator = 'euler',
        policy_save_path = 'data/',
        media_save_path = 'data/training/',
    ):


    times = np.arange(Ti, Tf, Ts)
    nsteps = len(times)
    quad_params = get_quad_params()
    ctrl_params = get_ctrl_params()
    mlp_state_dict = torch.load(policy_save_path + 'nav_policy.pth')
    x0 = quad_params["default_init_state_np"]
    R = reference.waypoint('wp_p2p', average_vel=1.0)

    # Safety Filter Parameters
    # ------------------------
    nx = 17             # number of states
    nu = 4
    h_offset = 0.12     # the height above zero of the terminal SF CBF
    xr_setpoint = R(1)  # the setpoint about which we will be navigating
    P = np.eye(nx)      # the matrix multiplying the quadratic term in the terminal SF CBF

    # convert the dynamics into "time varying" form
    f = lambda x, k, u: state_dot.casadi_vectorized(x, u, quad_params)

    options = {
        'event-trigger': False
    }

    sf_params = {
        'N': N_sf,              # prediction horizon
        'dt': Ts,
        'delta': 0.000005,      # Small robustness margin for constraint tightening
        'alpha2': 1.0,          # Multiplication factor for feasibility parameters
        'integrator_type': 'Euler',
        'nx': nx,
        'nu': nu
        }

    state_constraints = {}
    input_constraints = {}

    # State constraint functions: g(x) <= 0
    kk = 0
    for ii in range(nx):
        state_constraints[kk] = lambda x, k, xmax=quad_params['state_ub'], ind=ii: x[ind] - xmax[ind]
        kk += 1
        state_constraints[kk] = lambda x, k, xmin=quad_params['state_lb'], ind=ii: xmin[ind] - x[ind]
        kk += 1

    # Input constraints: h(u) <= 0
    kk = 0
    for ii in range(nu):
        input_constraints[kk] = lambda u, ii=ii, umax=[quad_params['maxCmd']]*nu: u[ii] - umax[ii]
        kk += 1
        input_constraints[kk] = lambda u, ii=ii, umin=[quad_params['minCmd']]*nu: umin[ii] - u[ii]
        kk += 1

    constraints = {
        'hf': lambda x, k, xr=R(1), h_eps=h_offset: (x - xr).T @ P @ (x - xr) - h_offset,
        'state_constraints': state_constraints,
        'input_constraints': input_constraints
    }

    # Define the Neuromancer system
    # -----------------------------

    node_list = []

    def process_policy_input(x, r, c, radius=0.5):
        idx = [0,7,1,8,2,9]
        x_r, r_r = x[:, idx], r[:, idx]
        x_r = torch.clip(x_r, min=ptu.tensor(-3.), max=ptu.tensor(3.))
        r_r = torch.clip(r_r, min=ptu.tensor(-3.), max=ptu.tensor(3.))
        c_pos, c_vel = posVel2cyl(x_r, c, radius)
        return torch.hstack([r_r - x_r, c_pos, c_vel])
    process_policy_input_node = nm.system.Node(process_policy_input, ['X', 'R', 'Cyl'], ['Obs'], name='state_selector')
    node_list.append(process_policy_input_node)

    mlp = nm.modules.blocks.MLP(6 + 2, 3, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU,
                    hsizes=[20, 20, 20, 20]).to(ptu.device)
    policy_node = nm.system.Node(mlp, ['Obs'], ['MLP_U'], name='mlp')
    node_list.append(policy_node)

    gravity_offset = lambda u: u + ptu.tensor([[0,0,-quad_params["hover_thr"]]])
    gravity_offset_node = nm.system.Node(gravity_offset, ['MLP_U'], ['MLP_U_grav'], name='gravity_offset')
    node_list.append(gravity_offset_node)

    pid = PID(Ts=Ts, bs=1, ctrl_params=ctrl_params, quad_params=quad_params)
    pid_node = nm.system.Node(pid, ['X', 'MLP_U_grav'], ['U', 'PID_X'], name='pid_control')
    node_list.append(pid_node)
    
    safety_filter = SafetyFilter(x0=x0, f=f, params=sf_params, constraints=constraints, options=options)
    def filter(x, u, unom=):
        unom.pid.reset(pid_x)
    filter = lambda x, u: safety_filter.compute_control_step(unom, x, u).unsqueeze(0)
    filter_node = nm.system.Node(filter, input_keys=['X', 'U'], output_keys=['U_filtered'], name='dynamics')
    node_list.append(filter_node)

    sys = mujoco_quad(state=quad_params["default_init_state_np"], quad_params=quad_params, Ti=Ti, Tf=Tf, Ts=Ts, integrator=integrator)
    sys_node = nm.system.Node(sys, input_keys=['X', 'U_filtered'], output_keys=['X'], name='dynamics')
    node_list.append(sys_node)

    cl_system = nm.system.System(node_list, nsteps=nsteps)

    # load the pretrained policies
    cl_system.nodes[1].load_state_dict(mlp_state_dict)

    # set the mujoco simulation to the correct initial conditions
    cl_system.nodes[5].callable.set_state(np.copy(quad_params["default_init_state_np"]))

    # Define The Neuromancer Dataset
    # ------------------------------

    X = ptu.from_numpy(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32))
    # X[:,:,7] = 1.5 # xdot adversarial
    # X[:,:,8] = 1.5 # ydot adversarial

    # X[:,:,0] = 0.45
    # X[:,:,1] = 0.45
    data = {
        'X': X,
        'R': torch.concatenate([ptu.from_numpy(R(1)[None,:][None,:].astype(np.float32))]*nsteps, axis=1),
        'Cyl': torch.concatenate([ptu.tensor([[[1,1]]])]*nsteps, axis=1),
        'Violation': ptu.tensor([[[False]]*nsteps])
    }

    # Perform CLP Simulation
    # ----------------------

    start_time = time.time()
    output = cl_system.forward(data, retain_grad=False, print_loop=True)
    end_time = time.time()
    print(f'time taken for simulation: {end_time - start_time}')

    # Save and Plot
    # -------------

    # save
    print("saving the state and input histories...")
    x_history = np.stack(ptu.to_numpy(output['X'].squeeze()))
    u_history = np.stack(ptu.to_numpy(output['U'].squeeze()))
    r_history = np.stack(ptu.to_numpy(output['R'].squeeze()))

    np.savez(
        file = f"data/xu_sf_p2p_mj_{str(Ts)}.npz",
        x_history = x_history,
        u_history = u_history,
        r_history = r_history
    )

    animator = Animator(x_history, times, r_history, max_frames=500, save_path=media_save_path, state_prediction=None, drawCylinder=True)
    animator.animate()

    print('fin')

if __name__ == "__main__":

    # example usage
    import utils.pytorch as ptu
    from utils.integrate import euler, RK4

    ptu.init_dtype()
    ptu.init_gpu(use_gpu=False)

    Ti, Tf, Ts = 0.0, 3.5, 0.001
    N_sf = 2
    N_pred = 100
    Tf_hzn_sf, Tf_hzn_pred = 0.5, 0.1
    integrator = euler

    run_nav_no_event_trigger(Ti, Tf, Ts, N_sf, N_pred, Tf_hzn_sf, Tf_hzn_pred, integrator)

    quad_params = get_quad_params()

    safety_filter = SafetyFilter(state_dot.casadi_vectorized, Ts, Tf_hzn_sf, N_sf, quad_params, integrator)
    
    def example_usage():
        u_seq = np.array([[1., 1., 1., 1.]]*N_sf)
        x_seq = np.vstack([quad_params["default_init_state_np"]]*(N_sf+1))
        passthrough = safety_filter(u_seq, x_seq, True)
        computed = safety_filter(u_seq, x_seq, False)

    example_usage()

    

    print('fin')