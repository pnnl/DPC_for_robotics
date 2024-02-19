import casadi as ca
import l4casadi as l4c
import torch
import neuromancer as nm
import numpy as np
from tqdm import tqdm
from time import time

from utils.integrate import euler, RK4, generate_variable_timesteps
from dpc import posVel2cyl

from dynamics import get_quad_params, mujoco_quad
from utils.quad import plot_mujoco_trajectories_wp_p2p, \
        calculate_mpc_cost, Animator
from safe_set2 import SafeSet
from utils.geometry import find_closest_simplex_equation
import utils.pytorch as ptu
from pid import get_ctrl_params, PID
import reference

# Define a function to adjust the layer names
def adjust_layer_names(key):
    # Remove 'callable.' and replace 'linear' with 'hidden_layers'
    new_key = key.replace('callable.linear', 'hidden_layers')
    
    # Split the key to check for layer index adjustments
    parts = new_key.split('.')
    
    if parts[0] == 'hidden_layers':
        index = int(parts[1])  # Get the index of the hidden layer
        
        # Rename 'hidden_layers.0' to 'input_layer'
        if index == 0:
            new_key = new_key.replace('hidden_layers.0', 'input_layer')
        
        # Rename 'hidden_layers.4' to 'output_layer'
        elif index == 4:
            new_key = new_key.replace('hidden_layers.4', 'output_layer')
        
        # Adjust the indices of the hidden layers to start at 0 again
        else:
            new_key = f'hidden_layers.{index - 1}' + '.' + '.'.join(parts[2:])
    
    return new_key

class DPCDynamics:
    def __init__(self) -> None:

        A = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

        B = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        self.f = lambda x, u: A @ x + B @ u

    def __call__(self, x, u):
        return self.f(x,u)

class SafetyFilter:

    def __init__(self, dynamics, dpc, Ts, Tf_hzn, N, quad_params, integrator) -> None:

        """ 
        dynamics    - the quad dynamics themselves in casadi
        dpc        - the casadi representation of the learned DPC controller
        Ts          - timestep
        Tf_hzn      - the predictive horizon final time
        N           - the number of steps in the predictive horizon (variable in length according to generate_variable_timesteps)
        quad_params - the literal quad parameters
        """

        dts = generate_variable_timesteps(Ts, Tf_hzn, N)
        self.dpc = dpc
        self.nx = 6
        self.nu = 3
        self.nvc = 2

        # margins for the hyperplane constraints to push within safe set
        m1 = 0.1
        m2 = 0.1

        # Define Box Constraints
        # ----------------------
        umax = 100
        umin = -100

        self.opti = ca.Opti()

        # Optimisation variables
        self.X      = self.opti.variable(self.nx, N + 1)
        self.U      = self.opti.variable(self.nu, N)

        # Time-varying parameters
        self.X0      = self.opti.parameter(self.nx, 1)
        self.w1      = self.opti.parameter(6, 1)         # normal vector for hyperplane constraints
        self.b1      = self.opti.parameter(1, 1)         # offset for hyperplane constraints
        self.w2      = self.opti.parameter(2, 1)         # normal vector for hyperplane constraints
        self.b2      = self.opti.parameter(1, 1)         # offset for hyperplane constraints
        self.Unom    = self.opti.parameter(3, 1)         # constant unom 
        self.Ujac    = self.opti.parameter(3, 8)         # unom jacobian for taylor approximation

        # Set up variables to warm start the optimization problems
        self.X_warm = np.zeros((self.nx, N+1))
        self.U_warm = np.zeros((self.nu, N))

        # Define casadi optimization parameters
        self.p_opts = {"expand": True, "print_time": False, "verbose": True}
        self.s_opts = {"max_iter": 5000, "print_level": 1, "tol": 1e-4}

        # Build Constraints
        # -----------------

        # Dynamics constraints
        for k in range(N):
            self.opti.subject_to(self.X[:, k + 1] == integrator(dynamics, self.X[:, k], self.U[:, k], dts[k]))
        
        # Initial condition constraints
        self.opti.subject_to(self.X[:,0] == self.X0)

        # Input constraints
        for i in range(self.nu):
            self.opti.subject_to(self.U[i] <= umax)
            self.opti.subject_to(self.U[i] >= umin)

        # Define Cost
        # -----------

        # function to go from true X to X that DPC expects
        Y, pv = self.X_to_DPC_input(self.X)

        # ReLU = lambda x: x # ca.fmax(x, 0)
        # ReLU = lambda x, alpha=0.1: ca.fmax(x, 0) + alpha * ca.fmin(x, 0) # leaky relu
        ReLU = lambda x: ca.log(1 + ca.exp(x)) # softplus

        x_sym = ca.MX.sym('x', 8, 1)
        y_sym = self.dpc(x_sym)
        self.Ujac_func = ca.Function('df', [x_sym], [ca.jacobian(y_sym, x_sym)])
        # x = ca.DM([[0.], [2.], [0.], [4], [0.], [2.], [0.], [4]])

        cost = 0.0
        for i in range(N):
            U_i_taylor = self.Unom + self.Ujac @ (Y[:,i] - Y[:,0])
            cost += 0.01 * ca.dot(U_i_taylor - self.U[:,i], U_i_taylor - self.U[:,i]) \
                    + 20 * ReLU(pv[:,i].T @ self.w2 + self.b2 + m2) \
                    + 1.5 * ReLU(self.X[:,i].T @ self.w1 + self.b1 + m1) 
        # 0.005, 10, 1 works
        # 0.01, 20, 1.5 best
        self.opti.minimize(cost)

        self.opti.set_initial(self.X, self.X_warm)
        self.opti.set_initial(self.U, self.U_warm)

        self.opti.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend

        # perform test solve
        self.opti.set_value(self.X0, np.array([2.,1.,3.,1.,2.,1.]))
        self.opti.set_value(self.w1, np.array([1.,1.,1.,1.,1.,1.]))
        self.opti.set_value(self.b1, -1)
        self.opti.set_value(self.w2, np.array([1.,1.]))
        self.opti.set_value(self.b2, -1)
        x = ca.DM([[0.], [2.], [0.], [4], [0.], [2.], [0.], [4]])
        self.opti.set_value(self.Unom, self.dpc(x))
        self.opti.set_value(self.Ujac, self.Ujac_func(x))

        self.opti.solve()

        print('init completed successfully')

    def X_to_DPC_input(self, X):
        pv = posVel2cyl.casadi(X, np.array([[1,1]]).T, 0.5)
        return ca.vertcat(X, pv), pv
    
    def __call__(self, e, obs, eqn1, eqn2):
        
        e = ptu.to_numpy(e).T
        obs = ptu.to_numpy(obs).T
        x = obs[:6]
        pv = obs[6:]

        print(f"dist2cyl_center: {pv[0]}")

        # x = ptu.to_numpy(x.T)
        eqn1 = ptu.to_numpy(eqn1).flatten()
        eqn2 = ptu.to_numpy(eqn2).flatten()

        pv = posVel2cyl.numpy(x, np.array([[1,1]]).T, 0.5)

        in_cvx_hull = eqn1[:-1] @ x + eqn1[-1] <= 0.2
        print(f"in_cvx_hull: {in_cvx_hull}")
        in_noncvx_hull = eqn2[:-1] @ pv + eqn2[-1] <= 0
        print(f"in_noncvx_hull: {in_noncvx_hull}")

        y = ca.DM(e)
        unom = self.dpc(y).full()

        near_ball = np.linalg.norm(e[:6], ord=2) <= 0.5
        print(f"near_ball: {near_ball}")

        if (in_cvx_hull and in_noncvx_hull) or near_ball:

            return ptu.from_numpy(unom[:,0]), ptu.from_numpy(self.X_warm)
        else:
            if x[0,0] == 1. and x[2,0] == 1.:
                x[0:2,0] += 1e-3 # avoids being exactly in center of cylinder - confuses cylindrical coordinates

            self.opti.set_value(self.Unom, unom)
            self.opti.set_value(self.Ujac, self.Ujac_func(y))

            self.opti.set_value(self.X0, x)
            self.opti.set_value(self.w1, eqn1[:-1])
            self.opti.set_value(self.b1, eqn1[-1])
            self.opti.set_value(self.w2, eqn2[:-1])
            self.opti.set_value(self.b2, eqn2[-1])

            self.opti.set_initial(self.X, self.X_warm)
            self.opti.set_initial(self.U, self.U_warm)

            sol = self.opti.solve()

            u = ptu.from_numpy(np.array(sol.value(self.U[:,0])).reshape((self.nu,1))[:,0])
            print(f"u delta: {ptu.from_numpy(unom).flatten() - u}")

            return u, ptu.from_numpy(self.X_warm)
        
def run(Ti, Tf, Ts):

    quad_params = get_quad_params()
    ctrl_params = get_ctrl_params()

    mlp_state_dict = torch.load('data/nav_policy.pth')
    log = torch.load('large_data/nav_training_data.pt')
    times = np.arange(Ti, Tf, Ts)
    nsteps = len(times)
    ss = SafeSet(log)
    R = reference.waypoint('wp_p2p', average_vel=1.0)
    
    N_sf = 30 # 30
    Tf_hzn_sf = 0.5 # 0.50
    dpc_dynamics = DPCDynamics()
    mlp = l4c.naive.MultiLayerPerceptron(6+2,20,3,4,'ReLU')
    mlp.load_state_dict(l4c_state_dict)
    unom = l4c.L4CasADi(mlp, model_expects_batch_dim=True)
    safety_filter = SafetyFilter(dpc_dynamics, unom, Ts, Tf_hzn_sf, N_sf, quad_params, euler)

    node_list = []

    def process_policy_input(x, r, c, radius=0.5):
        idx = [0,7,1,8,2,9]
        x_r, r_r = x[:, idx], r[:, idx]
        x_r = torch.clip(x_r, min=ptu.tensor(-3.), max=ptu.tensor(3.))
        r_r = torch.clip(r_r, min=ptu.tensor(-3.), max=ptu.tensor(3.))
        c_pos, c_vel = posVel2cyl.pytorch_vectorized(x_r, c, radius)
        return torch.hstack([r_r - x_r, c_pos, c_vel]), torch.hstack([x_r, c_pos, c_vel])
    process_policy_input_node = nm.system.Node(process_policy_input, ['X', 'R', 'Cyl'], ['ObsErr_CylPV', 'Obs_CylPV'], name='state_selector')
    node_list.append(process_policy_input_node)

    mlp = nm.modules.blocks.MLP(6 + 2, 3, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU,
                    hsizes=[20, 20, 20, 20]).to(ptu.device)
    mlp.load_state_dict({k.replace("callable.", ""): v for k, v in mlp_state_dict.items()})
    policy_node = nm.system.Node(mlp, ['ObsErr_CylPV'], ['MLP_U'], name='mlp')
    node_list.append(policy_node)

    def nearest_halfspace(obs, ss=ss):
        cvx_hull_point = ptu.to_numpy(obs[:,:6])
        cvx_eqn = find_closest_simplex_equation(ss.cvx_hull, cvx_hull_point)
        non_cvx_hull_point = np.hstack(posVel2cyl.numpy_vectorized(cvx_hull_point, np.array([[1.,1.]]), 0.5)).flatten() 
        non_cvx_eqn = find_closest_simplex_equation(ss.non_cvx_hull, non_cvx_hull_point)
        
        # cvx_eqn = np.array([0.,0.,0.,0.,0.,0,-100]) # check to see if theres an interaction between the safe sets
        # non_cvx_eqn = np.array([0., 1., -100]) 
        return ptu.from_numpy(cvx_eqn[None,:]), ptu.from_numpy(non_cvx_eqn[None,:])
    nearest_halfspace_node = nm.system.Node(nearest_halfspace, input_keys=['Obs_CylPV'], output_keys=['Halfspace_cvx', 'Halfspace_noncvx'])
    node_list.append(nearest_halfspace_node)

    def filter(e, x, equation_cvx, equation_noncvx):
        u, x_planned = safety_filter(e, x, equation_cvx, equation_noncvx)
        return u.unsqueeze(0), x_planned
    filter_node = nm.system.Node(filter, input_keys=['ObsErr_CylPV', 'Obs_CylPV', 'Halfspace_cvx', 'Halfspace_noncvx'], output_keys=['MLP_U_filtered', 'X_planned'], name='dynamics')
    node_list.append(filter_node)

    gravity_offset = lambda u: u + ptu.tensor([[0,0,-quad_params["hover_thr"]]])
    gravity_offset_node = nm.system.Node(gravity_offset, ['MLP_U_filtered'], ['MLP_U_filtered_grav'], name='gravity_offset')
    node_list.append(gravity_offset_node)

    pid = PID(Ts=Ts, bs=1, ctrl_params=ctrl_params, quad_params=quad_params)
    pid_node = nm.system.Node(pid, ['X', 'MLP_U_filtered_grav'], ['U', 'PID_X'], name='pid_control')
    node_list.append(pid_node)

    sys = mujoco_quad(state=quad_params["default_init_state_np"], quad_params=quad_params, Ti=Ti, Tf=Tf, Ts=Ts, integrator=euler)
    sys_node = nm.system.Node(sys, input_keys=['X', 'U'], output_keys=['X'], name='dynamics')
    node_list.append(sys_node)

    cl_system = nm.system.System(node_list, nsteps=nsteps)

    npoints = 5  # Number of points you want to generate in 2 dimensions ie. 5 == (5x5 grid)
    xy_values = ptu.from_numpy(np.linspace(-1, 1, npoints))  # Generates 'npoints' values between -10 and 10 for x
    z_values = ptu.from_numpy(np.linspace(-1, 1, 1))
    z_values = [ptu.tensor(0.)]

    datasets = []
    for xy in xy_values:
        v = 2.25 # directly towards the cylinder
        xr = 1
        yr = 1
        x = xy
        y = -xy
        vy = v * torch.sin(torch.arctan((yr-y)/(xr-x)))
        vx = v * torch.cos(torch.arctan((yr-y)/(xr-x)))
        for z in z_values:
            datasets.append({
                'X': ptu.tensor([[[xy,-xy,z,1,0,0,0,vx,vy,0,0,0,0,*[522.9847140714692]*4]]]),
                'R': torch.concatenate([ptu.from_numpy(R(1)[None,:][None,:].astype(np.float32))]*nsteps, axis=1),
                'Cyl': ptu.tensor([[[1,1]]*nsteps]),
            })

    # Perform CLP Simulation
    outputs = []
    start_time = time()
    for data in tqdm(datasets):
        # set the mujoco simulation to the correct initial conditions
        cl_system.nodes[6].callable.set_state(ptu.to_numpy(data['X'].squeeze().squeeze()))
        cl_system.nodes[5].callable.reset(None)
        outputs.append(cl_system.forward(data, retain_grad=False))

    end_time = time()
    total_time = (end_time - start_time)
    average_time = total_time / npoints

    print("Average Time: {:.2f} seconds".format(average_time))
    x_histories = [ptu.to_numpy(outputs[i]['X'].squeeze()) for i in range(npoints)]
    u_histories = [ptu.to_numpy(outputs[i]['U'].squeeze()) for i in range(npoints)]
    r_histories = [np.vstack([R(1)]*(nsteps+1))]*npoints

    plot_mujoco_trajectories_wp_p2p(outputs, 'data/paper/dpc_adv_nav.svg')

    average_cost = np.mean([calculate_mpc_cost(x_history, u_history, r_history) for (x_history, u_history, r_history) in zip(x_histories, u_histories, r_histories)])

    print("Average MPC Cost: {:.2f}".format(average_cost))


    print('fin')

if __name__ == "__main__":

    mlp_state_dict = torch.load('data/nav_policy.pth')

    # Create a new state dict with the modified keys
    l4c_state_dict = {adjust_layer_names(key): value for key, value in mlp_state_dict.items()}

    # Load into l4casadi model
    mlp = l4c.naive.MultiLayerPerceptron(6+2,20,3,4,'ReLU')
    mlp.load_state_dict(l4c_state_dict)
    unom = l4c.L4CasADi(mlp, model_expects_batch_dim=True)

    quad_params = get_quad_params()
    ctrl_params = get_ctrl_params()

    Ti, Tf, Ts = 0.0, 5.0, 0.001
    N_sf = 30
    N_pred = None
    Tf_hzn_sf, Tf_hzn_pred = 0.50, None # 0.75 works, is best that does
    dpc_dynamics = DPCDynamics()

    # state_dot.casadi_vectorized

    safety_filter = SafetyFilter(dpc_dynamics, unom, Ts, Tf_hzn_sf, N_sf, quad_params, euler)
    
    x = ptu.tensor([[2.,1.,3.,1.,2.,1.,1.,1.]])
    eqn1 = ptu.tensor([[1.,1.,1.,1.,1.,1.,-1.]])
    eqn2 = ptu.tensor([[1.,1.,-1.]])

    safety_filter(x, x, eqn1, eqn2)

    Ti, Tf, Ts = 0., 5.0, 0.001
    run(Ti, Tf, Ts)




