"""
From DADAIST/DPC/Safety_Filter/safetyFilter.py by Wenceslao Shaw Cortez
"""
from tqdm import tqdm
from casadi import *
import numpy as np
import traceback
from dpc_sf.dynamics.eom_pt import QuadcopterPT
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.control.trajectory.trajectory import waypoint_reference

class SafetyFilter():
    def __init__(self, x0, f, params, constraints, options=None):

        # Initialize terms
        self.N = params['N']                                        # Prediction horizon
        self.delta = params['delta']                                # Constraint tightening parameter
        self.dT = params['dt']                                      # Sampling time
        self.integrator_type = params['integrator_type']            # String dictating type of integrators e.g., Euler

        self.nx = params['nx']                                      # Number of states
        self.nu = params['nu']                                      # Number of control inputs
        self.hf = constraints['hf']                                 # Function de/Users/vilj909/Code/dpc/safety_filter/building_safety_filter.pyfining the terminal set (sublevel set of hf)
        self.state_constraints = constraints['state_constraints']   # Dictionary holding all state constraints
        self.input_constraints = constraints['input_constraints']   # Dictionary holding all input constraints
        self.nc = len(self.state_constraints)                       # Number of state constraints
        self.x0 = x0                                                # Initial condition
        self.nu = params['nu']                                      # Number of inputs
        self.ncu = len(self.input_constraints)                      # Number of input constraints
        self.f = f                                                  # System dynamics
        self.options = options                                      # Dictionary with additional options

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
        # the alpha_f - the real alpha
        if 'alpha' not in params:
            self.alpha = 10000.0  # Gain term required to be large as per Wabersich2022 paper
        else:
            self.alpha = params['alpha']

        # the alpha on the non terminal slack terms, which the paper sets to 1
        if 'alpha2' not in params:
            self.alpha2 = 1.0
        else:
            self.alpha2 = params['alpha2']

        # Define optimization variables and constraints for the feasibility problem and safety filter problem
        self.setup_opt_variables()
        self.setup_constraints(self.x0, 0)

        # Set up variables to warm start the optimization problems
        self.U_warm = np.zeros((self.nu, self.N))
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

    def setup_opt_variables(self):
        '''Set up the optimization variables and related terms'''

        # Define optimization in casadi for safety filter and feasibility problems
        self.opti = Opti()
        self.opti_feas = Opti()

        # State, input variables for safety filter
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
        self.dynamics_constraints(X=self.X, U=self.U, opt=self.opti)
        self.dynamics_constraints(X=self.X_f, U=self.U_f, opt=self.opti_feas)

        # Define state , input, constraints for the feasibility problem
        self.define_constraints(X=self.X_f, U=self.U_f, opt=self.opti_feas, x0 = x0, k0=k0, Xi=self.Xi_f, XiN=self.XiN_f)

        # Define non-negativity constraints for slack terms of the feasibility problem
        [self.opti_feas.subject_to(self.Xi_f[ii,:] >= 0.0) for ii in range(self.nc)]
        self.opti_feas.subject_to(self.XiN_f >= 0.0)

        # Define casadi optimization parameters
        self.p_opts = {
            "expand": True, 
            "print_time": 0, 
            "ipopt.print_level": 0, 
            "verbose": False,
            'ipopt.tol': 1e-6   
        }
        self.s_opts = {} #{ 'max_iter': 5000}
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

    def dynamics_constraints(self, X, U, opt):
        '''Defines the constraints related to the ode of the system dynamics
        X: system state
        U: system input
        opt: Casadi optimization class'''
        if self.integrator_type == 'RK4':
            # Runge-Kutta 4 integration
            for k in range(self.N):  # loop over control intervals (these are the equality constraints associated with the dynamics)
                k1 = self.f(X[:, k], 0, U[:, k])
                k2 = self.f(X[:, k] + self.dT / 2 * k1, 0, U[:, k])
                k3 = self.f(X[:, k] + self.dT / 2 * k2, 0, U[:, k])
                k4 = self.f(X[:, k] + self.dT * k3, 0, U[:, k])
                x_next = X[:, k] + self.dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                opt.subject_to(X[:, k + 1] == x_next)  # close the gaps

        if self.integrator_type == 'Euler':
            # Standard Euler integration
            for k in range(self.N):  # loop over control intervals (these are the equality constraints associated with the dynamics)
                x_next = X[:, k] + self.dT * self.f(X[:, k], 0, U[:, k])
                opt.subject_to(X[:, k + 1] == x_next)  # close the gaps

        if self.integrator_type == 'cont':
            # Treat system as 'continuous'
            for k in range(self.N):  # loop over control intervals (these are the equality constraints associated with the dynamics)
                x_next = self.f(X[:, k], 0, U[:, k])
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
                # TODO: -kk * self.delta is the constraint tightening
                opt.subject_to(self.state_constraints[ii](X[:, kk],kk+k0) <= -kk*self.delta + Xi[ii,kk-1] - self.rob_marg)

        # Terminal constraint, enforced at last time step
        opt.subject_to(self.hf(X[:, -1]) <= XiN -self.rob_marg_term)

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
        is_feasible, x_pred, u_pred = self.check_opl_feas(unom=unom, x0=x, N=100)
        # is_feasible = False

        if is_feasible is True:
            print("nominal feasible")
            u = unom(x, k)
        else:
            print("nominal not feasible")
            # Set up the optimization variables, constraints for the current state and time step
            self.setup_opt_variables()
            self.setup_constraints(x0=x, k0=k)

            # Solve the safety filter, only take the optimal control trajectory, and return the first instance of the control trajectory
            sol = self.solve_safety_filter(unom, x, k)[0]
            u = np.array(sol.value(self.U[:,0])).reshape((self.nu,1))
        return u

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
        else:
            Xi_sol = np.zeros((self.nc, self.N))
            XiN_sol = 0.0

        # Store solution to warm start next problem
        self.U_warm = feas_sol.value(self.U_f)
        self.Xi_warm = Xi_sol
        self.XiN_warm = XiN_sol

        # print('Safety filter slack terms:')
        # print('Xi_f = ' + str(Xi_sol))
        # print('Xi_N = ' + str(XiN_sol))
        # print('U_feas = '+str(feas_sol.value(self.U_f)))

        # Define constraints for the feasibility problem and control objective to remain minimally close to unom
        self.define_constraints(X=self.X, U=self.U, opt=self.opti, x0=x0, k0=k0, Xi=Xi_sol, XiN=XiN_sol )
        self.opti.minimize(dot(self.U[:, 0] - unom(x0, k0),self.U[:, 0] - unom(x0, k0) )) #self.opti.minimize((self.U[:, 0] - unom(x0, k0)).T @ (self.U[:, 0] - unom(x0,k0)))

        # Warm start the optimization
        self.opti.set_initial(self.U, self.U_warm)

        # Set the solver
        self.opti.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend

        return self.opti.solve(), Xi_sol, XiN_sol

    def solve_feasibility(self):
        '''Solve the feasibility problem'''

        # Define the objetive to penalize the use of slack variables
        self.opti_feas.minimize(self.alpha * self.XiN_f + self.alpha2*sum([self.Xi_f[:, kk].T @ self.Xi_f[:, kk] for kk in range(self.N-1)]))

        # Warm start the optimization
        self.opti_feas.set_initial(self.U_f, self.U_warm)
        self.opti_feas.set_initial(self.Xi_f, self.Xi_warm)
        self.opti_feas.set_initial(self.XiN_f, self.XiN_warm)

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

        else:
            self.quad.reset(x0.full()[:,0])
            x = np.copy(x0.full()[:,0])

        u_pred = []
        x_pred = []
        for k in range(N):
            u = unom(x, k)
            self.quad.step(u)
            u_pred.append(u)
            x_pred.append(x)
            x = self.quad.get_state()

        lb = quad_params["state_lb"]
        ub = quad_params["state_ub"]

        for x in x_pred:
            if not np.all(x[:13] >= lb[:13]) or not np.all(x[:13] <= ub[:13]):
                return False, x_pred, u_pred
            # Cylinder constraint check
            r = 0.5
            x_c = 1.0
            y_c = 1.0
            if r**2 - (x[0] - x_c)**2 - (x[1] - y_c)**2 > 0:  # If this is positive, constraint is violated
                return False, x_pred, u_pred
        return True, x_pred, u_pred
    
            

class ClosedLoopSystem():
    '''Closed-loop simulation of given dynamics f with control u'''

    def __init__(self, f, u, u_nom, dt, int_type='Euler'):
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

    def simulate(self, x0, N):
        '''Simulates the closed-loop system starting from state x0 for N steps'''

        # Initialize states
        x = x0
        #print('x = ' + str(x))
        nx = len(x)
        X = x.flatten()
        #print('X = ' + str(X))


        # Iterate through simulation horizon
        for kk in tqdm(range(N)):
            # Compute control and integrate to compute the next state
            # uk = self.u.compute_control_step(self.u_nom, x, kk)
            try:
                uk = self.u.compute_control_step(self.u_nom, x, kk)
            except:
                print('Time: k = '+str(kk))
                print(traceback.format_exc())
                print('WARNING: Control computation failed, exiting closed-loop simulation')
                return X,U
            #print('uk = '+str(uk))
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

        return X, U

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

    def compute_control_step(self, unom, x, k):
        '''Given unom, x, k output the unom value evaluated at x,k ie. unom(x,k)'''
        return unom(x, k)
