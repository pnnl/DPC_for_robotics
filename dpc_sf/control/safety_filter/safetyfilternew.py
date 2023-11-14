from casadi import *
import numpy as np
from tqdm import tqdm
import traceback
from dpc_sf.dynamics.eom_pt import QuadcopterPT
from dpc_sf.dynamics.eom_ca import QuadcopterCA
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.control.trajectory.trajectory import waypoint_reference
from dpc_sf.utils import pytorch_utils as ptu
import copy
import torch

class SafetyFilter():
    def __init__(self, x0, f, params, constraints, options=None):

        # Initialize terms
        self.N = params['N']                                        # Prediction horizon
        self.delta = params['delta']                                # Constraint tightening parameter
        self.dT = params['dt']                                      # Sampling time
        self.integrator_type = params['integrator_type']            # String dictating type of integrators e.g., Euler

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
        if 'robust_margin' not in params:
            self.rob_marg = 0.0
        else:
            self.rob_marg = params['robust_margin']
        if 'robust_margin_terminal' not in params:
            self.rob_marg_term = 0.0
        else:
            self.rob_marg_term = params['robust_margin_terminal']

        # If alpha is not included, set to default value
        if 'alpha' not in params:
            self.alpha = 10000.0  # Gain term required to be large as per Wabersich2022 paper
        else:
            self.alpha = params['alpha']
        if 'alpha2' not in params:
            self.alpha2 = 1.0
        else:
            self.alpha2 = params['alpha2']

        # Define optimization variables and constraints for the feasibility problem and safety filter problem
        self.setup_opt_variables(initial_setup=True)
        self.setup_constraints(self.x0, 0)

        # Set up variables to warm start the optimization problems
        self.X_warm = np.zeros((self.nx, self.N+1))
        self.U_warm = np.zeros((self.nu, self.N))
        self.Xf_warm = np.zeros((self.nx, self.N+1))
        self.Uf_warm = np.zeros((self.nu, self.N))
        self.Xi_warm = np.zeros((self.nc, self.N-1))
        self.XiN_warm = 0.0

        # John setup quad for open loop checking
        # --------------------------------------
        self.quad = QuadcopterPT(
            state=quad_params["default_init_state_np"],
            reference=waypoint_reference('wp_p2p', average_vel=1.0),
            params=quad_params,
            Ts=self.dT,
            Ti=0,
            Tf=self.N * self.dT,
            integrator='euler',
        )

    def setup_opt_variables(self, initial_setup=False):
        '''Set up the optimization variables and related terms'''

        # Delete previous optimization setup
        #if not initial_setup:
        #    del self.opti
        #    del self.opti_feas

        # Define optimization in casadi for safety filter and feasibility problems
        self.opti = Opti()
        self.opti_feas = Opti()

        # Stete, input variables for safety filter
        self.X = self.opti.variable(self.nx, self.N + 1)
        self.U = self.opti.variable(self.nu, self.N)

        # State, slack, input variables for  feasibility problem
        self.X_f = self.opti_feas.variable(self.nx, self.N + 1)
        self.Xi_f = self.opti_feas.variable(self.nc, self.N-1 )
        self.XiN_f = self.opti_feas.variable()
        self.U_f = self.opti_feas.variable(self.nu, self.N)

    def setup_constraints(self, x0, k0):
        '''Set up constraints '''

        # Define differential equation/difference equation equality constraint for safety filter and feasibility problem
        self.dynamics_constraints(X=self.X, U=self.U, opt=self.opti, k0=k0)
        self.dynamics_constraints(X=self.X_f, U=self.U_f, opt=self.opti_feas, k0=k0)

        # Define state , input, constraints for the feasibility problem
        self.define_constraints(X=self.X_f, U=self.U_f, opt=self.opti_feas, x0=x0, k0=k0, Xi=self.Xi_f, XiN=self.XiN_f)

        # Define non-negativity constraints for slack terms of the feasibility problem
        [self.opti_feas.subject_to(self.Xi_f[ii,:] >= 0.0) for ii in range(self.nc)]
        self.opti_feas.subject_to(self.XiN_f >= 0.0)

        # Define casadi optimization parameters
        self.p_opts = {
            "expand": True, 
            "print_time": 0, 
            "verbose": False,
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-6
        }
        self.s_opts = { 'max_iter': 300, #'linear_solver':'mumps', 'mumps_scaling':8,
                        'warm_start_init_point': 'yes'}
        # self.s_opts = {  # "print_level": 0,
        #    "tol": 1e-5,
        #    "dual_inf_tol": 1e-5,
        #    "constr_viol_tol": 1e-5,
        #    "compl_inf_tol": 1e-5,
        #    "acceptable_tol": 1e-5,
        #    "acceptable_constr_viol_tol": 1e-5,
        #    "acceptable_dual_inf_tol": 1e-5,
        #    "acceptable_compl_inf_tol": 1e-5,
        #    "acceptable_obj_change_tol": 1e20,
        #    "diverging_iterates_tol": 1e20,
        #    "nlp_scaling_method": "none"}

    def dynamics_constraints(self, X, U, opt, k0=0):
        '''Defines the constraints related to the ode of the system dynamics
        X: system state
        U: system input
        opt: Casadi optimization class'''
        if self.integrator_type == 'RK4':
            # Runge-Kutta 4 integration
            for k in range(self.N):  # loop over control intervals (these are the equality constraints associated with the dynamics)
                k1 = self.f(X[:, k], k0 + k, U[:, k])
                k2 = self.f(X[:, k] + self.dT / 2 * k1, k0 + k + self.dT / 2, U[:, k])
                k3 = self.f(X[:, k] + self.dT / 2 * k2, k0 + k + self.dT / 2, U[:, k])
                k4 = self.f(X[:, k] + self.dT * k3, k0 + k + self.dT, U[:, k])
                x_next = X[:, k] + self.dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                opt.subject_to(X[:, k + 1] == x_next)  # close the gaps

        if self.integrator_type == 'Euler':
            # Standard Euler integration
            for k in range(self.N):  # loop over control intervals (these are the equality constraints associated with the dynamics)
                x_next = X[:, k] + self.dT * self.f(X[:, k], k0 + k, U[:, k])
                opt.subject_to(X[:, k + 1] == x_next)  # close the gaps

        if self.integrator_type == 'cont':
            # Treat system as 'continuous'
            for k in range(self.N):  # loop over control intervals (these are the equality constraints associated with the dynamics)
                x_next = self.f(X[:, k], k0 + k, U[:, k])
                opt.subject_to(X[:, k + 1] == x_next)  # close the gaps

    def define_constraints(self, X, U, opt, x0, k0=0, Xi=None, XiN=None):
        '''Defines the system constraints, i.e, state and input constraints on the system
        X: system state
        U: system input
        opt: Casadi optimization class
        x0: initial state
        k0: initial time step
        Xi: optional slack terms for the system constraints
        XiN: optional slack term for terminal constraint'''

        # If no slack terms are passed, set them to zero
        if (Xi is None) or (XiN is None):
            Xi = np.zeros((self.nc, self.N))
            XiN = 0.0

        # Trajectory constraints,
        for ii in range(self.nx):
            # initial condition constraint
            opt.subject_to(X[ii, 0] == x0[ii])

        # State trajectory constraints, enforced at each time step
        for ii in range(self.nc):
            for kk in range(1,self.N):
                opt.subject_to(self.state_constraints[ii](X[:, kk],kk+k0) <= -kk*self.delta + Xi[ii,kk-1] - self.rob_marg)

        # Terminal constraint, enforced at last time step
        opt.subject_to(self.hf(X[:, -1]) <= XiN - self.rob_marg_term)

        # Input constraints
        for ii in range(self.ncu):
            for kk in range(self.N):
                opt.subject_to(self.input_constraints[ii](U[:, kk]) <= 0.0)

    def compute_control_step(self, unom, x, k):
        ''' Compute the safety filter and output the control for the next time instant
        unom: Given nominal control to be implemented at current time if constraints are satisfied
        unom: nominal control unom(x,k)
        x: current state
        k: current time step
        '''

        # predict whether or not the nominal control will breach the constraints
        is_feasible, x_pred, u_pred = self.check_opl_feas(unom=unom, x0=x, N=self.N)
        # x_pred, u_pred = None, None
        # is_feasible = True
        if is_feasible is True:
            print("nominal feasible")
            u = unom(x, k)
        
        else:
            # Set up the optimization variables, constraints for the current state and time step
            self.setup_opt_variables()
            self.setup_constraints(x0=x, k0=k)

            # Solve the safety filter, only take the optimal control trajectory, and return the first instance of the control trajectory
            sol = self.solve_safety_filter(unom, x, k)[0]

            # Save solution for warm start
            self.X_warm = sol.value(self.X)
            self.U_warm = sol.value(self.U)
            try:
                self.lamg_warm = sol.value(self.opti.lam_g)
            except:
                print('initialized, no dual variables')
            # print('Safety filter terms:')
            # print('X = ' + str(self.X_warm))
            # print('U = ' + str(self.U_warm))

            u = np.array(sol.value(self.U[:,0])).reshape((self.nu,1))
        return u, x_pred, u_pred

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
            Xi_sol = feas_sol.value(self.Xi_f)
            XiN_sol = feas_sol.value(self.XiN_f)

            # Save solutions for warm start
            self.Xf_warm = feas_sol.value(self.X_f)
            self.Uf_warm = feas_sol.value(self.U_f)
            self.Xi_warm = Xi_sol
            self.XiN_warm = XiN_sol
            try:
                self.lamgf_warm = feas_sol.value(self.opti_feas.lam_g) # dual variables
            except:
                print('initialized no dual variables yet for feasibility problem')

            # print('Safety filter feasibility terms:')
            # print('Xi_f = ' + str(Xi_sol))
            # print('Xi_N = ' + str(XiN_sol))
            # print('X = '+str(feas_sol.value(self.X_f)))
            # print('U_feas = ' + str(feas_sol.value(self.U_f)))
        else:
            Xi_sol = np.zeros((self.nc, self.N))
            XiN_sol = 0.0

        # Define constraints for the feasibility problem and control objective to remain minimally close to unom
        self.define_constraints(X=self.X, U=self.U, opt=self.opti, x0=x0, k0=k0, Xi=Xi_sol, XiN=XiN_sol )
        self.opti.minimize(dot(self.U[:, 0] - unom(x0, k0),self.U[:, 0] - unom(x0, k0) )) #self.opti.minimize((self.U[:, 0] - unom(x0, k0)).T @ (self.U[:, 0] - unom(x0,k0)))

        # Warm start the optimization
        #self.opti.set_initial(self.X, self.X_warm)
        #self.opti.set_initial(self.U, self.U_warm)
        self.opti.set_initial(self.X, self.Xf_warm)
        self.opti.set_initial(self.U, self.Uf_warm)
        #try:

        #self.opti.set_initial(self.opti.lam_g, self.lamg_warm)

        self.opti.set_initial(self.opti.lam_g, self.lamgf_warm[:-((self.N-1)*self.nc+1)]) # Removing lambdas associated with non-negative constraint in feasibility problem
        #except:
            #print('initialized no dual variables yet for safety filter problem')

        # Set the solver
        self.opti.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend
        try:
            return self.opti.solve(), Xi_sol, XiN_sol
        except:
            print(traceback.format_exc())
            print('------------------------------------INFEASIBILITY---------------------------------------------')
            self.infeasibilities.append(k0)
            return self.opti.debug, Xi_sol, XiN_sol


    def solve_feasibility(self):
        '''Solve the feasibility problem'''

        # Define the objetive to penalize the use of slack variables
        self.opti_feas.minimize(self.alpha * self.XiN_f + self.alpha2*sum([self.Xi_f[:, kk].T @ self.Xi_f[:, kk] for kk in range(self.N-1)]))

        # Warm start the optimization
        self.opti_feas.set_initial(self.X_f, self.Xf_warm)
        self.opti_feas.set_initial(self.U_f, self.Uf_warm)
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
        self.setup_opt_variables()
        self.setup_constraints(x0=x0, k0=k0)
        sol, Xi_sol, XiN_sol = self.solve_safety_filter(unom, x0, k0)
        return sol.value(self.X), sol.value(self.U), Xi_sol, XiN_sol

    def check_opl_feas(self, unom, x0, N):
        # check that the states of the rollout of the pytorch quad
        # from x0 under unom remain within bounds for N steps
        print("checking unfiltered feasibility")
        if type(x0) is numpy.ndarray:
            self.quad.reset(x0)
            x = np.copy(x0)
        elif type(x0) is torch.Tensor:
            self.quad.reset(ptu.to_numpy(x0))
            x = np.copy(x0)
        elif type(x0) is DM:
            self.quad.reset(x0.full()[:,0])
            x = np.copy(x0.full()[:,0])
        else:
            raise Exception(f"unexpected input type: {type(x0)}")

        u_pred = []
        x_pred = []
        for k in range(N):
            u = unom(x, k)
            self.quad.step(u)
            u_pred.append(np.copy(u))
            x_pred.append(np.copy(x))
            x = self.quad.get_state()

        lb = quad_params["state_lb"]
        ub = quad_params["state_ub"]

        # return True, x_pred, u_pred

        for i, x in enumerate(x_pred):
            if not np.all(x[:13] >= lb[:13]) or not np.all(x[:13] <= ub[:13]):
                print("VIOLATION: state constraint predicted to be violated")
                print(f"violation occured {i} timesteps in the future")
                return False, x_pred, u_pred
            # Cylinder constraint check
            r = 0.51
            x_c = 1.0
            y_c = 1.0
            if r**2 - (x[0] - x_c)**2 - (x[1] - y_c)**2 > 0:  # If this is positive, constraint is violated
                print("VIOLATION: cylinder constraint predicted to be violated")
                print(f"violation occured {i} timesteps in the future")
                return False, x_pred, u_pred
        print("NOMINAL: no constraints predicted to be violated")
        return True, x_pred, u_pred
    

class ClosedLoopSystem():
    '''Closed-loop simulation of given dynamics f with control u'''

    def __init__(self, f, u, u_nom, dt, int_type='Euler', use_true_unom=True):
        '''
        f: Function defining system dynamics f(x,k,u)
        u: Control that closes the loop in the system, must have member compute_control_step(unom, x,k)
        unom: Nominal control, function of x and k, unom(x,k)
        dt: Sampling time for Euler integration
        '''
        self.f = f
        self.u = u
        self.u_nom = u_nom
        self.dt = dt
        self.int_type = int_type
        self.use_true_unom = use_true_unom

    def simulate(self, x0, N):
        '''Simulates the closed-loop system starting from state x0 for N steps'''

        # Initialize states
        x = x0
        #print('x = ' + str(x))
        nx = len(x)
        X = x.flatten()
        #print('X = ' + str(X))

        x_pred_history = []
        u_pred_history = []
        # Iterate through simulation horizon
        for kk in tqdm(range(N)):
            # Compute control and integrate to compute the next state
            # uk, x_pred, u_pred = self.u.compute_control_step(self.u_nom, x, kk)

            # try:
            print('Time: k = ' + str(kk))
            if self.use_true_unom is True:
                uk, x_pred, u_pred = self.u.compute_control_step(self.u_nom, x, kk)
            elif self.use_true_unom is False:
                # first get memory of future unoms to pull from
                is_feasible, x_pred, u_pred_nom = self.u.check_opl_feas(self.u_nom, x0=x, N=150)

                is_feasible = True

                if kk <= 10: # give a few timesteps at the start to allow the 
                    policy_change_counter = 0
                    last_iter_feas = True
                    is_feasible = True

                if last_iter_feas is False:
                    policy_change_counter = 10
                elif last_iter_feas is True:
                    if policy_change_counter > 0:
                        policy_change_counter -= 1
                    elif policy_change_counter <= 0:
                        policy_change_counter = 0

                if (is_feasible is True) and (policy_change_counter == 0):
                    # uk, x_pred, u_pred = self.u.compute_control_step(self.u_nom, x, kk)
                    print("is feasible, and policy_change_counter = 0")
                    uk = self.u_nom(x, kk)
                    last_iter_feas = True

                elif ((is_feasible is False) and (kk != 0)) or ((is_feasible is True) and (policy_change_counter > 0)):

                    print(f"policy_change_counter: {policy_change_counter}")

                    if last_iter_feas is True:
                        print("last iteration was feasible, retaining")
                        feas_counter = 0
                        last_u_pred = np.stack(copy.deepcopy(last_u_pred_nom))[:,:,0]
                        

                    elif last_iter_feas is False:
                        print("last iteration infeasible, reusing unom memory")
                        feas_counter += 1
                        print(f"steps since last unom update: {feas_counter}")

                    else:
                        print("unexpected eventuality")

                    print(f"feas_counter used for this timestep: {feas_counter}")
                    unom = lambda x, k: last_u_pred[feas_counter]
                    uk, _, u_pred = self.u.compute_control_step(unom, x, kk)

                if is_feasible is False:
                    last_iter_feas = False
                elif is_feasible is True and policy_change_counter == 0:
                    last_iter_feas = True

                elif is_feasible is False and kk == 0:
                    print("first timestep of simulation")
                    test = lambda x, k: u_pred_nom[0]
                    uk, _, u_pred = self.u.compute_control_step(test, x, kk)
                    

                last_u_pred_nom = copy.deepcopy(u_pred_nom)


            x_pred_history.append(x_pred)
            # u_pred_history.append(u_pred)
            # except:

            #     print(traceback.format_exc())
            #     print('WARNING: Control computation failed, exiting closed-loop simulation')
            #     return X,U
            print('uk = '+str(uk))
            print(f'x = {str(x[0])}')
            print(f'y = {str(x[1])}')
            print(f'z = {str(x[2])}')
            print(f'x_dot = {str(x[0+7])}')
            print(f'y_dot = {str(x[1+7])}')
            print(f'z_dot = {str(x[2+7])}')
            print(f'dist2cyl = {str(np.sqrt((x[0]-1)**2+(x[1]-1)**2) - 0.5)}')
            
            x = self.integrator(self.f(x, kk, uk), x)

            # For the first time step, get the size of u and setup U to collect all control inputs
            if kk == 0:
                try:
                    nu = len(uk)
                except:
                    nu = 1
                    uk = np.array([uk])
                U = uk.reshape((1, nu))
            if nu == 1:
                uk = np.array([uk])
            #print('uk = '+str(uk))
            # Store all states and variables
            #print('X = '+str(X))
            #print('x = '+str(x))
            X = np.vstack((X, np.array(x[:,0]).flatten()))
            U = np.vstack((U, uk.reshape((1, nu))))
            #print('U = '+str(U))

        if self.u.infeasibilities:
            print('Infeasibilities of safety filter occured at the following time steps: '+str(self.u.infeasibilities))
        else:
            print('No infeasibilities occurred.')

        return X, U, x_pred_history, u_pred_history

    def integrator(self, f, x):
        '''Integration of dynamics f at state x'''
        #print('f = '+str(f))
        if self.int_type == 'Euler':
            return x + f * self.dt
        if self.int_type == 'cont':
            return f


class DummyControl:
    '''Used to create same control class as in the safety_filter for an arbitrary control'''

    def __init__(self):
        self.name = 'dummy'
        self.infeasibilities = []

    def compute_control_step(self, unom, x, k):
        '''Given unom, x, k output the unom value evaluated at x,k ie. unom(x,k)'''
        return unom(x, k)

if __name__ == "__main__":
    ### basic parameters
    nx = 17
    nu = 4
    N = 30     # prediction horizon No. steps
    Nsim = 300  # simulation length

    Ts = 0.01
    Tf_hzn = 3.0
    N = 300
    Ti = 0.0
    Tf = 15.0
    integrator_type = "Euler"
    obstacle_opts = {'r': 0.5, 'x': 1, 'y': 1}
    # Find the optimal dts for the MPC
    dt_1 = Ts
    d = (2 * (Tf_hzn/N) - 2 * dt_1) / (N - 1)
    dts = [dt_1 + i * d for i in range(N)]
    ### dummy policy
    policy = lambda x, t: np.array([0.,0.,0.,0.])
    ### terminal constraint
    xr = np.copy(quad_params["default_init_state_np"])
    xr[0] = 2 # x  
    xr[1] = 2 # y
    xr[2] = 1 # z flipped
    ### create dynamics
    quad = QuadcopterCA()
    def f(x, t, u):
        return quad.state_dot_1d(x, u)[:,0]
    ### define SF parameters
    params = {}
    params['N'] = N  # prediction horizon
    params['Ts'] = Ts  # sampling time
    params['dts'] = dts
    params['delta'] = 0.00005  # Small robustness margin for constraint tightening
    params['robust_margin'] = 0.1 #0.0 #  #   Robustness margin for handling perturbations
    params['robust_margin_terminal'] = 0.0001 #0.0 # # Robustness margin for handling perturbations (terminal set)
    params['alpha2'] = 1.0  # Multiplication factor for feasibility parameters
    params['integrator_type'] = integrator_type  # Integrator type for the safety filter
    params['nx'] = nx
    params['nu'] = nu
    ### System constraints
    terminal_constraint = {}
    state_constraints = {}
    input_constraints = {}
    # state upper/lower bounds
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
    def cylinder_constraint(x, k):
        current_time = sum1(dts[:k])
        multiplier = 1 + current_time * 0.1
        current_x, current_y = x[0], x[1]
        return r ** 2 * multiplier - (current_x - x_c)**2 - (current_y - y_c)**2
    state_constraints[2] = cylinder_constraint# lambda x, k: r**2 * (1 + k * dt * 0.001) - (x[0] - x_c)**2 - (x[1] - y_c)**2
    # Quadratic CBF (assuming linearisation valid)
    cbf_const = (np.sqrt(2) - 0.5) ** 2
    hf = lambda x: (x-xr).T @ (x-xr) - cbf_const
    # hf = lambda x: r**2 - (x[0] - x_c)**2 - (x[1] - y_c)**2
    # input constraints
    kk = 0
    nu = 4
    for ii in range(nu):
        input_constraints[kk] = lambda u: u[ii] - umax[0]
        kk += 1
        input_constraints[kk] = lambda u: umin[0] - u[ii]
        kk += 1    
    # combine constraints to send to SF
    constraints = {}
    constraints['state_constraints'] = state_constraints
    constraints['hf'] = hf
    constraints['input_constraints'] = input_constraints    
    ### instantiate safety filter
    x0 = quad_params["default_init_state_np"]
    SF = SafetyFilter(x0=x0, f=f, params=params, constraints=constraints, options={})
    # CL = sf.ClosedLoopSystem(f=f_pert, u=self.SF, u_nom=self.unom, dt=dt, int_type='Euler', use_true_unom=True)
    u = SF(x0, np.array([10.,0.,-10.,0.]))
    print('fin')