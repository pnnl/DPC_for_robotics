import traceback
import numpy as np
import torch
import neuromancer as nm
from neuromancer.dynamics import integrators
import casadi as ca
import copy

import utils.pytorch as ptu
from dpc import posVel2cyl
from utils.integrate import euler, RK4, generate_variable_timesteps, generate_variable_times
from dynamics import get_quad_params, mujoco_quad, state_dot
from pid import get_ctrl_params, PID
import reference
from utils.quad import Animator

class SafetyFilter:
    def __init__(self, x0, f, params, constraints, options=None):

        # Initialize terms
        self.N = params['N']                                        # Prediction horizon
        self.delta = params['delta']                                # Constraint tightening parameter
        self.dT = params['dt']                                      # Sampling time
        self.integrator_type = params['integrator_type']            # String dictating type of integrators e.g., Euler
        self.integrator = globals()[self.integrator_type].time_variant.numpy

        self.nx = params['nx']                                      # Number of states
        self.nu = params['nu']                                      # Number of control inputs
        try:
            self.ny = params['ny']                                  # Number of outputs
        except:
            self.ny = self.nx
        self.hf = constraints['hf']                                 # Function defining the terminal set (sublevel set of hf)
        self.state_constraints = constraints['state_constraints']   # Dictionary holding all state constraints
        self.input_constraints = constraints['input_constraints']   # Dictionary holding all input constraints
        self.nc = len(self.state_constraints)                       # Number of state constraints
        self.x0 = x0                                                # Initial condition
        self.nu = params['nu']                                      # Number of inputs
        self.ncu = len(self.input_constraints)                      # Number of input constraints
        self.f = f                                                  # System dynamics
        self.options = options                                      # Dictionary with additional options

        self.infeasibilities = []

        # If use_feas has not been set, set to default of True
        if 'use_feas' not in options:
            self.options['use_feas'] = True

        # If no robustness margin is set, set it to zero
        if 'robust_margins' not in params:
            self.rob_marg = {'Lh_x': 0, 'Ld': 0, 'Lf_x': 0, 'Lhf_x':0}
        else:
            self.rob_marg = params['robust_margins']

        # If no event-triggering is set, it is set to true
        if 'event-trigger' not in options:
            self.options['event-trigger'] = True
        self.events = []
        self.events_log = {}
        self.slacks = []
        self.slacks_term = []

        # If alpha is not included, set to default value
        if 'alpha' not in params:
            self.alpha = 10000.0  # Gain term required to be large as per Wabersich2022 paper
        else:
            self.alpha = params['alpha']
        if 'alpha2' not in params:
            self.alpha2 = 1.0
        else:
            self.alpha2 = params['alpha2']

        # 'time-varying' needs to be set to true if the system/constraints are time-varying
        # This requires the optimization objects from casadi needs to be reset for every control computation
        if 'time-varying' not in options:
            self.options['time-varying'] = True

        # Define optimization variables and constraints for the feasibility problem and safety filter problem
        self.setup_opt_variables()
        self.setup_constraints()

        # Set up variables to warm start the optimization problems
        self.X_warm = np.zeros((self.nx, self.N+1))
        self.U_warm = np.zeros((self.nu, self.N))
        self.Xf_warm = np.zeros((self.nx, self.N + 1))
        self.Uf_warm = np.zeros((self.nu, self.N))
        if self.N > 1: # Only use intermediate slack variables if prediction is more than 1-step lookahead
            self.Xi_warm = np.zeros((self.nc, self.N))
        self.XiN_warm = 0.0


    def setup_opt_variables(self):
        '''Set up the optimization variables and related terms'''

        # Define optimization in casadi for safety filter and feasibility problems
        self.opti_feas = ca.Opti()
        self.opti = ca.Opti()

        # State, slack, input variables for  feasibility problem
        self.X0_f = self.opti_feas.parameter(self.nx, 1)
        self.X_f = self.opti_feas.variable(self.nx, self.N + 1)
        if self.N > 1:
            self.Xi_f = self.opti_feas.variable(self.nc, self.N )
        self.XiN_f = self.opti_feas.variable()
        self.U_f = self.opti_feas.variable(self.nu, self.N)

        # State, input variables for safety filter
        self.X0 = self.opti.parameter(self.nx,1)
        self.X = self.opti.variable(self.nx, self.N + 1)
        self.U = self.opti.variable(self.nu, self.N)
        if self.N > 1:
            self.Xi = self.opti.parameter(self.nc, self.N )
        self.XiN = self.opti.parameter()


    def setup_constraints(self, k0=0):
        '''Set up constraints
         k0: initial time
         '''

        # Define differential equation/difference equation equality constraint for safety filter and feasibility problem
        self.dynamics_constraints(X=self.X_f, U=self.U_f, opt=self.opti_feas, k0=k0)
        self.dynamics_constraints(X=self.X, U=self.U, opt=self.opti, k0=k0)

        # Define state , input, constraints
        if self.N > 1:
            self.define_constraints(X=self.X_f, U=self.U_f, opt=self.opti_feas, x0=self.X0_f, k0=k0, Xi=self.Xi_f, XiN=self.XiN_f)
            self.define_constraints(X=self.X, U=self.U, opt=self.opti, x0=self.X0, k0=k0, Xi=self.Xi, XiN=self.XiN)
        if self.N == 1:
            self.define_constraints_1step(X=self.X_f, U=self.U_f, opt=self.opti_feas, x0=self.X0_f, k0=k0, XiN=self.XiN_f)
            self.define_constraints_1step(X=self.X, U=self.U, opt=self.opti, x0=self.X0, k0=k0, XiN=self.XiN)

        # Define non-negativity constraints for slack terms of the feasibility problem
        if self.N > 1:
            [self.opti_feas.subject_to(self.Xi_f[ii,:] >= 0.0) for ii in range(self.nc)]
        self.opti_feas.subject_to(self.XiN_f >= 0.0)

        # Define casadi optimization parameters
        self.p_opts = {"expand": True, "print_time": False, "verbose": False}
        self.s_opts = { 'max_iter': 300,
                        'print_level': 1,
                        'warm_start_init_point': 'yes',
                        'tol': 1e-5,
                        # 'constr_viol_tol': 1e-8,
                        # "compl_inf_tol": 1e-8,
                        # "acceptable_tol": 1e-4,
                        # "acceptable_constr_viol_tol": 1e-8,
                        # "acceptable_dual_inf_tol": 1e-8,
                        # "acceptable_compl_inf_tol": 1e-8,
                        }


    def dynamics_constraints(self, X, U, opt, k0=0):
        '''Defines the constraints related to the ode of the system dynamics
        X: system state
        U: system input
        opt: Casadi optimization class
        k0: initial time
        '''

        # Loop over control intervals (these are the equality constraints associated with the dynamics)
        for k in range(self.N):
            
            x_next = self.integrator(self.f, X[:, k], k0 + k, U[:, k], self.dT)
            opt.subject_to(X[:, k + 1] == x_next)


    def define_constraints(self, X, U, opt, x0, k0=0, Xi=None, XiN=None):
        '''Defines the system constraints, i.e, state and input constraints on the system
        X: system state
        U: system input
        opt: Casadi optimization class
        x0: initial state
        k0: initial time step
        Xi: optional slack terms for the system constraints
        XiN: optional slack term for terminal constraint'''

        # Trajectory constraints,
        for ii in range(self.nx):
            # initial condition constraint
            opt.subject_to(X[ii, 0] == x0[ii])

        # State trajectory constraints, enforced at each time step
        for ii in range(self.nc):
            for kk in range(0,self.N):
                rob_marg = self.compute_robust_margin(kk)
                opt.subject_to(self.state_constraints[ii](X[:, kk],kk+k0) <= Xi[ii,kk] - rob_marg)

        # Terminal constraint, enforced at last time step
        rob_marg_term = self.compute_robust_margin_terminal()
        opt.subject_to(self.hf(X[:, -1], k0+self.N) <= XiN - rob_marg_term)

        # Input constraints
        for ii in range(self.ncu):
            for kk in range(self.N):
                opt.subject_to(self.input_constraints[ii](U[:, kk]) <= 0.0)


    def define_constraints_1step(self, X, U, opt, x0, k0=0, XiN=None):
        '''Defines the system constraints, i.e, state and input constraints on the system
        X: system state
        U: system input
        opt: Casadi optimization class
        x0: initial state
        k0: initial time step
        XiN: optional slack term for terminal constraint'''

        # Trajectory constraints,
        for ii in range(self.nx):
            # initial condition constraint
            opt.subject_to(X[ii, 0] == x0[ii])

        # Terminal constraint, enforced at last time step (NOTE: here hf is time-varying)
        rob_marg_term = self.compute_robust_margin_terminal()
        opt.subject_to(self.hf(X[:, -1], k0 + 1) <= XiN - rob_marg_term)

        # Input constraints
        for ii in range(self.ncu):
            opt.subject_to(self.input_constraints[ii](U) <= 0.0)


    def check_constraint_satisfaction(self, unom, x0, k0=0):
        '''Check if constraints are satisfied for the future trajectory for given nominal control
         at the state x0 at time k0
         unom: function nominal control, unom(x,k)
         x0: initial state
         k0: initial time
         '''

        # Loop over control intervals (these are the equality constraints associated with the dynamics)
        x = x0.T
        for k in range(self.N):

            # Integrate system
            u = unom(x.T, k + k0)
            try:
                u = unom(x.T, k + k0)
            except:
                return 1, 'unom evaluation failed'
            x = self.integrator(self.f, x, k0 + k, u, self.dT)

            # Check input constraints
            for iu in range(self.ncu):
                check_input = self.input_constraints[iu](np.array(u).reshape(self.nu,1))
                if check_input > 0.0:
                    return 1, 'input_constraint_violation'

            # Check state constraints (TO DO: add a check before the for loop to check initial condition is in safe set)
            for ic in range(self.nc):
                rob_marg = self.compute_robust_margin(k)
                check_states = self.state_constraints[ic](x, k + k0) + rob_marg
                # Element-wise comparison with 0.0
                comparison = check_states > 0.0

                # Sum the boolean values (True is treated as 1, False as 0)
                sum_comparison = ca.sum1(comparison)

                if sum_comparison > 0.0:
                    return 1, 'state_constraint_violation'

        # Check terminal constraint
        rob_marg_term = self.compute_robust_margin_terminal()
        check_term = self.hf(x, k + k0 + 1.0) + rob_marg_term
        if check_term > 0.0:
            print("terminal constraint violated, ignoring...")
            # return 1, 'terminal_constraint_violation'

        return 0, 'no_violation'


    def check_constraint_satisfaction_1step(self, unom, x0, k0=0):
        '''Check if constraints are satisfied for the 1 step look ahead for given nominal control
         at the state x0 at time k0
         unom: function nominal control, unom(x,k)
         x0: initial state
         k0: initial time
         '''

        # Check input constraints
        for iu in range(self.ncu):
            u = unom(x0, k0)
            check_input = self.input_constraints[iu](np.array(u).reshape(self.nu,1))
            if check_input > 0.0:
                return 1, 'input_constraint_violation'

        # Check event condition
        hf = self.hf(x0, k0)
        if hf < -self.rob_marg['a']:
            return 0, 'no_violation'
        else:
            return 1, 'terminal_contraint_violation'


    def compute_robust_margin(self, kk):
        '''Given dictionary of robustness margin terms, compute the robustness margin
        kk: Current time step
        '''
        if kk > 0.0:
            return kk*self.delta + self.rob_marg['Lh_x']*self.rob_marg['Ld']*sum([ self.rob_marg['Lf_x']**j for j in range(kk) ])
        else:
            return 0.0


    def compute_robust_margin_terminal(self):
        '''Given dictionary of robustness margin terms, compute the robustness margin for the terminal constraint
        '''
        return self.rob_marg['Lhf_x']*self.rob_marg['Ld']*self.rob_marg['Lf_x']**(self.N-1)


    def compute_control_step2(self, u_seq, x_seq, k, constraints_satisfied):

        if self.options['event-trigger'] and constraints_satisfied:
            # Save solution for warm start
            self.X_warm = x_seq
            self.U_warm = u_seq
            return u_seq[:,0]
        else:
            # Set up the optimization variables, constraints for the current state and time step
            # if the system is time-varying
            if self.options['time-varying']:
                self.setup_opt_variables()
                self.setup_constraints(k0=k)

            # Set values to initial condition parameters
            self.opti_feas.set_value(self.X0_f, x_seq[:,0])
            self.opti.set_value(self.X0, x_seq[:,0])

            # Solve the safety filter, only take the optimal control trajectory, and return the first instance of the control trajectory
            sol = self.solve_safety_filter(unom=u_seq, x=x_seq[:,0], k=k)[0]

            # Save solution for warm start
            self.X_warm = sol.value(self.X)
            self.U_warm = sol.value(self.U)
            try:
                self.lamg_warm = sol.value(self.opti.lam_g)
            except:
                print('initialized, no dual variables')

            return np.array(sol.value(self.U[:,0])).reshape((self.nu,1))[:,0]

    def compute_control_step(self, unom, x, k):
        ''' Compute the safety filter and output the control for the next time instant
        unom: Given nominal control to be implemented at current time if constraints are satisfied
        unom: nominal control unom(x,k)
        x: current state
        k: current time step
        '''

        # Check event-triggering, if no event triggered, return unom(x,k)
        if self.options['event-trigger']:
            if self.N > 1:
                event = self.check_constraint_satisfaction(unom, x, k)
            if self.N == 1:
                event = self.check_constraint_satisfaction_1step(unom, x, k)
            self.events.append(event[0])
            self.events_log[k] = event[1]
            if not event[0]:
                return unom(x, k)[:,0]

        # Set up the optimization variables, constraints for the current state and time step
        # if the system is time-varying
        if self.options['time-varying']:
            self.setup_opt_variables()
            self.setup_constraints(k0=k)

        # Set values to initial condition parameters
        self.opti_feas.set_value(self.X0_f, x)
        self.opti.set_value(self.X0, x)

        # Solve the safety filter, only take the optimal control trajectory, and return the first instance of the control trajectory
        sol = self.solve_safety_filter(unom, x, k)[0]

        # Save solution for warm start
        self.X_warm = sol.value(self.X)
        self.U_warm = sol.value(self.U)
        try:
            self.lamg_warm = sol.value(self.opti.lam_g)
        except:
            print('initialized, no dual variables')

        return np.array(sol.value(self.U[:,0])).reshape((self.nu,1))[:,0]


    def solve_safety_filter(self, unom, x0, k0=0):
        '''Solve the safety filter optimization problem
        unom: Given nominal control to be implemented at current time if constraints are satisfied
        unom: nominal control unom(x,k)
        x0: initial state
        k0: initial time step
        '''

        # Solve feasibility problem, if use_feas is True, then use slack terms, else set them to zero
        if self.options['use_feas']:
            feas_sol = self.solve_feasibility()
            if self.N > 1:
                Xi_sol = feas_sol.value(self.Xi_f)
                self.slacks.append(np.linalg.norm(Xi_sol))
            XiN_sol = feas_sol.value(self.XiN_f)
            self.slacks_term.append( XiN_sol )

            # Save solutions for warm start
            self.Xf_warm = feas_sol.value(self.X_f)
            self.Uf_warm = feas_sol.value(self.U_f)
            if self.N > 1:
                self.Xi_warm = Xi_sol
            self.XiN_warm = XiN_sol
            try:
                self.lamgf_warm = feas_sol.value(self.opti_feas.lam_g) # dual variables
            except:
                print('initialized no dual variables yet for feasibility problem')

        else:
            if self.N > 1:
                Xi_sol = np.zeros((self.nc, self.N))
            XiN_sol = 0.0

        # Set slack variables and control objective to remain minimally close to unom
        if self.N > 1:
            self.opti.set_value(self.Xi, Xi_sol)
        self.opti.set_value(self.XiN, XiN_sol)
        self.opti.minimize(ca.dot(self.U[:, 0] - unom[:,0],self.U[:, 0] - unom[:,0] ))

        # Warm start the optimization
        self.opti.set_initial(self.X, self.Xf_warm)
        self.opti.set_initial(self.U, self.Uf_warm)
        self.opti.set_initial(self.opti.lam_g, self.lamgf_warm[:-((self.N)*self.nc+1)]) # Removing lambdas associated with non-negative constraint in feasibility problem

        # Set the solver
        self.opti.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend
        try:
            if self.N > 1:
                return self.opti.solve(), Xi_sol, XiN_sol
            if self.N == 1:
                return self.opti.solve(), XiN_sol
        except:
            print(traceback.format_exc())
            print('------------------------------------INFEASIBILITY---------------------------------------------')
            self.infeasibilities.append(k0)
            if self.N > 1:
                return self.opti.debug, Xi_sol, XiN_sol
            if self.N ==1 :
                return self.opti.debug, XiN_sol


    def solve_feasibility(self):
        '''Solve the feasibility problem'''

        # Define the objective to penalize the use of slack variables
        if self.N > 1:
            self.opti_feas.minimize(self.alpha * self.XiN_f + self.alpha2*sum([self.Xi_f[:, kk].T @ self.Xi_f[:, kk] for kk in range(self.N)]))

        if self.N == 1:
            self.opti_feas.minimize(self.alpha * self.XiN_f)

        # Warm start the optimization
        self.opti_feas.set_initial(self.X_f, self.Xf_warm)
        self.opti_feas.set_initial(self.U_f, self.Uf_warm)
        if self.N > 1:
            self.opti_feas.set_initial(self.Xi_f, self.Xi_warm)
        self.opti_feas.set_initial(self.XiN_f, self.XiN_warm)
        try:
            self.opti_feas.set_initial(self.opti_feas.lam_g, self.lamgf_warm)
        except:
            print('initialized no dual variables yet for feasibility problem')

        # Define the solver
        self.opti_feas.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend

        return self.opti_feas.solve()


    def open_loop_rollout(self, unom, x0, k0=0):
        '''Solve the safety filter problem and return the entire predicted state and control trajectories
        unom: Given nominal control to be implemented at current time if constraints are satisfied
        unom: nominal control unom(x,k)
        x0: initial state
        k0: initial time step
        '''

        # Set values to initial condition parameters
        self.opti_feas.set_value(self.X0_f, x0)
        self.opti.set_value(self.X0, x0)
        if self.N > 1:
            sol, Xi_sol, XiN_sol = self.solve_safety_filter(unom, x0, k0)
            return sol.value(self.X), sol.value(self.U), Xi_sol, XiN_sol

        if self.N == 1:
            sol, XiN_sol = self.solve_safety_filter(unom, x0, k0)
            return sol.value(self.X), sol.value(self.U), XiN_sol


    def clear_events(self):
        '''Clear event logs'''
        self.events = []
        self.events_log = {}


def get_sf_params(N_pred, Tf_hzn, Ts, quad_params, integrator):

    # terminal constraint set
    xr = np.copy(quad_params["default_init_state_np"])
    xr[0] = 2 # x  
    xr[1] = 2 # y
    xr[2] = 1 # z flipped

    # instantiate the dynamics
    # ------------------------
    quad = state_dot.casadi
    dt = Ts
    nx = 17
    f = lambda x, t, u: state_dot.casadi_vectorized(x, u, params=quad_params)

    # disturbance on the input
    # ------------------------
    disturbance_scaling = 0.0
    idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] # select all states to perturb
    pert_idx = [1 if i in idx else 0 for i in range(nx)]
    pert_mat = np.diag(pert_idx)
    d = lambda x, t, u: pert_mat * np.sin(t) * disturbance_scaling
    f_pert = lambda x, t, u: f(x, t, u) + d(x, t, u)

    # Safety Filter Parameters
    # ------------------------
    sf_params = {}
    sf_params['N'] = N_pred  # prediction horizon
    sf_params['dt'] = dt  # sampling time

        # Find the optimal dts for the MPC
    dt_1 = Ts
    d = (2 * (Tf_hzn/N_pred) - 2 * dt_1) / (N_pred - 1)
    dts_init = [dt_1 + i * d for i in range(N_pred)]

    sf_params['dts'] = dts_init
    sf_params['delta'] = 0.00005  # Small robustness margin for constraint tightening
    sf_params['robust_margin'] = 0.1 #0.0 #  #   Robustness margin for handling perturbations
    sf_params['robust_margin_terminal'] = 0.0001 #0.0 # # Robustness margin for handling perturbations (terminal set)
    sf_params['alpha2'] = 1.0  # Multiplication factor for feasibility parameters
    sf_params['integrator_type'] = 'euler'  # Integrator type for the safety filter

    # Define System Constraints
    # -------------------------
    terminal_constraint = {}
    state_constraints = {}
    input_constraints = {}

    sf_params['nx'] = 17
    sf_params['nu'] = 4

    umax = [quad_params["maxCmd"]]*100  #model.umax*np.ones((nu,1)) # Input constraint parameters(max)
    umin = [quad_params["minCmd"]]*100 #model.umin*np.ones((nu,1)) # Input constraint parameters (min)

    scu_k = quad_params["state_ub"] 
    scl_k = quad_params["state_lb"]
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
    hf = lambda x, k=0: (x-xr).T @ (x-xr) - cbf_const

    # Input constraints: h(u) <= 0
    kk = 0
    nu = 4
    for ii in range(nu):
        input_constraints[kk] = lambda u: u[ii] - umax[0]
        kk += 1
        input_constraints[kk] = lambda u: umin[0] - u[ii]
        kk += 1    

    constraints = {}
    constraints['state_constraints'] = state_constraints
    constraints['hf'] = hf
    constraints['input_constraints'] = input_constraints    

    # Implementation
    # --------------
    # Set up the safety filter and nominal control
    # x0 = random_state()[:,None] # 0.9*model.x0.reshape((nx, 1)) #
    x0 = quad_params["default_init_state_np"]

    return x0, f, sf_params, constraints, {'event-trigger': True}


def get_nominal_control_system_nodes(quad_params, ctrl_params, Ts):

    nominal_control_node_list = []

    def process_policy_input(x, r, c, radius=0.5):
        idx = [0,7,1,8,2,9]
        x_r, r_r = x[:, idx], r[:, idx]
        x_r = torch.clip(x_r, min=ptu.tensor(-3.), max=ptu.tensor(3.))
        r_r = torch.clip(r_r, min=ptu.tensor(-3.), max=ptu.tensor(3.))
        c_pos, c_vel = posVel2cyl(x_r, c, radius)
        return torch.hstack([r_r - x_r, c_pos, c_vel])
    process_policy_input_node = nm.system.Node(process_policy_input, ['X', 'R', 'Cyl'], ['Obs'], name='state_selector')
    nominal_control_node_list.append(process_policy_input_node)

    mlp = nm.modules.blocks.MLP(6 + 2, 3, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU,
                    hsizes=[20, 20, 20, 20]).to(ptu.device)
    policy_node = nm.system.Node(mlp, ['Obs'], ['MLP_U'], name='mlp')
    nominal_control_node_list.append(policy_node)

    gravity_offset = lambda u: u + ptu.tensor([[0,0,-quad_params["hover_thr"]]])
    gravity_offset_node = nm.system.Node(gravity_offset, ['MLP_U'], ['MLP_U_grav'], name='gravity_offset')
    nominal_control_node_list.append(gravity_offset_node)

    pid = PID(Ts=Ts, bs=1, ctrl_params=ctrl_params, quad_params=quad_params)
    pid_node = nm.system.Node(pid, ['X', 'MLP_U_grav'], ['U', 'PID_X'], name='pid_control')
    nominal_control_node_list.append(pid_node)

    return nominal_control_node_list


class Predictor(torch.nn.Module):

    def __init__(self, quad_params, ctrl_params, waypoint, Tf_hzn, N_pred, mlp_state_dict) -> None:
        super().__init__()
        # waypoint = R(1)
        self.r = torch.concatenate([ptu.from_numpy(waypoint[None,:][None,:])]*N_pred, axis=1)
        self.c = torch.concatenate([ptu.tensor([[[1,1]]])]*N_pred, axis=1)

        dts = generate_variable_timesteps(Ts, Tf_hzn, N_pred)

        node_list = get_nominal_control_system_nodes(quad_params, ctrl_params, Ts)

        predictive_dynamics = lambda x, u, k: (euler.time_invariant.pytorch(state_dot.pytorch_vectorized, x, u, dts[torch.round(k).long()], quad_params), k+1)
        predictive_dynamics_node = nm.system.Node(predictive_dynamics, input_keys=['X', 'U', 'K'], output_keys=['X', 'K'])
        node_list.append(predictive_dynamics_node)

        self.predictive_system = nm.system.System(node_list, nsteps=N_pred)
        self.predictive_system.nodes[1].load_state_dict(mlp_state_dict)
        # self.predictive_system.nodes[1].eval() # unecessary with torch.no_grad in the forward method.

        self.cylinder_center = ptu.tensor([1.,1.])
        self.cylinder_radius = 0.5

    def reset(self, pid_state):
        """
        The PID created for the low level control is stateful - the integrator, and some old
        values are retained for subsequent calculations. Therefore when using this control for
        testing the nominal control, we must reset the nominal control with the current state
        before every prediction rollout.
        """

        self.predictive_system.nodes[3].callable.reset(pid_state)

    def check_violations(self, x_pred):
        """
        See if the simulation yielded any problems with constraints
        """

        # Extract x and y coordinates
        x = x_pred[0, :, 1]
        y = x_pred[0, :, 2]

        # Calculate the squared distances from the cylinder center
        distances_squared = (x - self.cylinder_center[0])**2 + (y - self.cylinder_center[1])**2

        # Check if any point is inside the cylinder (distance <= radius)
        outside_cylinder = torch.any(distances_squared >= self.cylinder_radius**2).int()

        return outside_cylinder.unsqueeze(0).unsqueeze(0)

    def forward(self, x, pid_state):
        """
        Conduct a simulation
        """
        self.reset(pid_state)
        initial_conditions = {'X': x.unsqueeze(0), 'R':self.r, 'Cyl':self.c, 'K': ptu.tensor([[[0]]])}

        with torch.no_grad():
            predictions = self.predictive_system.forward(initial_conditions, retain_grad=False, print_loop=False)
        
        violations = self.check_violations(predictions['X'])

        return predictions['X'][0], predictions['U'][0], violations


def run_wp_p2p(
        Ti, Tf, Ts,
        N_pred, Tf_hzn,
        integrator = 'euler',
        policy_save_path = 'data/',
        media_save_path = 'data/training/',
    ):

    times = np.arange(Ti, Tf, Ts)
    nsteps = len(times)
    quad_params = get_quad_params()
    ctrl_params = get_ctrl_params()
    mlp_state_dict = torch.load(policy_save_path + 'wp_p2p_policy.pth')

    safety_filter = SafetyFilter(*get_sf_params(N_pred, Tf_hzn, Ts, quad_params, integrator))
    R = reference.waypoint('wp_p2p', average_vel=1.0)

    node_list = get_nominal_control_system_nodes(quad_params, ctrl_params, Ts)

    predictor = Predictor(quad_params, ctrl_params, R(1), Tf_hzn, N_pred, mlp_state_dict)
    predictor_node = nm.system.Node(predictor, input_keys=['X', 'PID_X'], output_keys=['X_pred', 'U_pred', 'Violation'], name='predictor')
    node_list.append(predictor_node)

    filter = lambda x_seq, u_seq, violation: ptu.from_numpy(safety_filter.compute_control_step2(ptu.to_numpy(u_seq.T), ptu.to_numpy(x_seq.T), 0, violation)).unsqueeze(0)
    filter_node = nm.system.Node(filter, input_keys=['X_pred', 'U_pred', 'Violation'], output_keys=['U_filtered'], name='dynamics')
    node_list.append(filter_node)

    sys = mujoco_quad(state=quad_params["default_init_state_np"], quad_params=quad_params, Ti=Ti, Tf=Tf, Ts=Ts, integrator=integrator)
    sys_node = nm.system.Node(sys, input_keys=['X', 'U_filtered'], output_keys=['X'], name='dynamics')
    node_list.append(sys_node)

    cl_system = nm.system.System(node_list, nsteps=nsteps)

    # load the pretrained policies
    cl_system.nodes[1].load_state_dict(mlp_state_dict)

    X = ptu.from_numpy(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32))
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
        file = f"data/xu_fig8_mj_{str(Ts)}.npz",
        x_history = x_history,
        u_history = u_history,
        r_history = r_history
    )

    animator = Animator(x_history, times, r_history, max_frames=500, save_path=media_save_path, state_prediction=None, drawCylinder=True)
    animator.animate()

    print('fin')

if __name__ == "__main__":

    ptu.init_dtype()
    ptu.init_gpu(use_gpu=False)

    Ti, Tf, Ts = 0.0, 3.5, 0.001
    N_pred = 100
    Tf_hzn = 0.1
    integrator = 'euler'

    run_wp_p2p(Ti, Tf, Ts, N_pred, Tf_hzn, integrator)

    print('fin')