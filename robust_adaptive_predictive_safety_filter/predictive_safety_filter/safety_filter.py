from casadi import *
import numpy as np
import traceback



class SafetyFilter():
    def __init__(self, x0, f, params, constraints, options=None):
        '''
        x0: (np.array size (nx,1)) Initial state condition
        f: (function, f(x,k,u)) Dynamics function, x_{k+1} = f(x_k, k, u_k)
        params: (dictionary) holds parmeters associated with the safety filter
        constraints: (dictionary) holds constraint functions, keys are:
                'state_constraints': (list of functions), holds list of state constraint functions of form: b(x,k) <= 0,
                'input_constraints': (list of functions) holds list of input constraint functions of form: U(u) <= 0,
                'hf': function, holds the terminal barrier function: hf(x,k) <= 0
        options: (dictionary) holds the following options for implementation:
                'use_feas': (bool) determines if additional feasibility optimization is used (see Wabersich et al 2023)
                'event-trigger': (bool) determines if event-triggering used or if safety filter solved at every time step
                'time-varying': (bool) if the system and constraints do not vary with time, set this to true so that the casadi optimization does not have to be re-setup at every time
        '''


        # Initialize terms
        self.N = params['N']                                        # Prediction horizon, must be > 1. Can only be =1 if using OneStepSafetyFilter
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

        # If alpha is not included, set to default value (only used in feasibility problem, if 'use_feas' set to true)
        if 'alpha' not in params:
            self.alpha = 10000.0  # Gain term required to be large as per Wabersich2023 paper
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
        self.initialize_warm_start_variables()


    def initialize_warm_start_variables(self):

        # Set up variables to warm start the optimization problems
        self.X_warm = np.zeros((self.nx, self.N + 1))
        self.U_warm = np.zeros((self.nu, self.N))
        self.Xf_warm = np.zeros((self.nx, self.N + 1))
        self.Uf_warm = np.zeros((self.nu, self.N))
        if self.N > 1:  # Only use intermediate slack variables if prediction is more than 1-step lookahead
            self.Xi_warm = np.zeros((self.nc, self.N))
        self.XiN_warm = 0.0

    def setup_opt_variables(self):
        '''Set up the optimization variables and related terms'''

        # Define optimization in casadi for safety filter and feasibility problems
        self.opti_feas = Opti()
        self.opti = Opti()

        # State, slack, input variables for  feasibility problem
        self.X0_f = self.opti_feas.parameter(self.nx, 1)
        self.X_f = self.opti_feas.variable(self.nx, self.N + 1)
        self.Xi_f = self.opti_feas.variable(self.nc, self.N )
        self.XiN_f = self.opti_feas.variable()
        self.U_f = self.opti_feas.variable(self.nu, self.N)

        # State, input variables for safety filter
        self.X0 = self.opti.parameter(self.nx,1)
        self.X = self.opti.variable(self.nx, self.N + 1)
        self.U = self.opti.variable(self.nu, self.N)
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
        self.define_constraints(X=self.X_f, U=self.U_f, opt=self.opti_feas, x0=self.X0_f, k0=k0, Xi=self.Xi_f, XiN=self.XiN_f)
        self.define_constraints(X=self.X, U=self.U, opt=self.opti, x0=self.X0, k0=k0, Xi=self.Xi, XiN=self.XiN)

        # Define non-negativity constraints for slack terms of the feasibility problem
        [self.opti_feas.subject_to(self.Xi_f[ii,:] >= 0.0) for ii in range(self.nc)]
        self.opti_feas.subject_to(self.XiN_f >= 0.0)

        # Define casadi optimization parameters
        self.p_opts = {"expand": True, "print_time": False, "verbose": False}
        self.s_opts = { 'max_iter': 300,
                        'print_level': 1,
                        'warm_start_init_point': 'yes',
                        'tol': 1e-5,
                        'constr_viol_tol': 1e-5,
                        "compl_inf_tol": 1e-5,
                        "acceptable_tol": 1e-5,
                        "acceptable_constr_viol_tol": 1e-5,
                        "acceptable_dual_inf_tol": 1e-5,
                        "acceptable_compl_inf_tol": 1e-5,
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
            x_next = integrate(self.f, X[:, k], k0 + k, U[:, k], self.dT, self.integrator_type)
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

    def check_constraint_satisfaction(self, unom, x0, k0=0):
        '''Check if constraints are satisfied for the future trajectory for given nominal control
         at the state x0 at time k0
         unom: function nominal control, unom(x,k)
         x0: initial state
         k0: initial time
         '''

        # Loop over control intervals (these are the equality constraints associated with the dynamics)
        x = x0
        for k in range(self.N):

            # Integrate system
            try:
                u = unom(x, k + k0)
            except:
                return 1, 'unom evaluation failed'
            x = integrate(self.f, x, k0 + k, u, self.dT, self.integrator_type)

            # Check input constraints
            for iu in range(self.ncu):
                check_input = self.input_constraints[iu](np.array(u).reshape(self.nu,1))
                if check_input > 0.0:
                    return 1, 'input_constraint_violation'

            # Check state constraints (TO DO: add a check before the for loop to check initial condition is in safe set)
            for ic in range(self.nc):
                rob_marg = self.compute_robust_margin(k)
                check_states = self.state_constraints[ic](x, k + k0) + rob_marg
                if check_states > 0.0:
                    return 1, 'state_constraint_violation'

        # Check terminal constraint
        rob_marg_term = self.compute_robust_margin_terminal()
        check_term = self.hf(x, k + k0 + 1.0) + rob_marg_term
        if check_term > 0.0:
            return 1, 'terminal_constraint_violation'

        return 0, 'no_violation'

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

    def compute_control_step(self, unom, x, k):
        ''' Compute the safety filter and output the control for the next time instant
        unom: Given nominal control to be implemented at current time if constraints are satisfied
        unom: nominal control unom(x,k)
        x: current state
        k: current time step
        '''

        # Check event-triggering, if no event triggered, return unom(x,k)
        if self.options['event-trigger']:
            event = self.check_constraint_satisfaction(unom, x, k)
            self.events.append(event[0])
            self.events_log[k] = event[1]
            if not event[0]:
                return unom(x, k)

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

        return np.array(sol.value(self.U[:,0])).reshape((self.nu,1))

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
            self.slacks.append(np.linalg.norm(Xi_sol))
            XiN_sol = feas_sol.value(self.XiN_f)
            self.slacks_term.append( XiN_sol )

            # Save solutions for warm start
            self.Xf_warm = feas_sol.value(self.X_f)
            self.X_warm = self.Xf_warm
            self.Uf_warm = feas_sol.value(self.U_f)
            self.U_warm = self.Uf_warm
            self.Xi_warm = Xi_sol
            self.XiN_warm = XiN_sol
            try:
                self.lamgf_warm = feas_sol.value(self.opti_feas.lam_g) # dual variables
            except:
                print('initialized no dual variables yet for feasibility problem')

        else:
            Xi_sol = np.zeros((self.nc, self.N))
            XiN_sol = 0.0

        # Set slack variables and control objective to remain minimally close to unom
        self.opti.set_value(self.Xi, Xi_sol)
        self.opti.set_value(self.XiN, XiN_sol)
        self.opti.minimize(dot(self.U[:, 0] - unom(x0, k0),self.U[:, 0] - unom(x0, k0) ))

        # Warm start the optimization
        self.opti.set_initial(self.X, self.X_warm)
        self.opti.set_initial(self.U, self.U_warm)
        if self.options['use_feas']:
            self.opti.set_initial(self.opti.lam_g, self.lamgf_warm[:-((self.N)*self.nc+1)]) # Removing lambdas associated with non-negative constraint in feasibility problem

        # Set the solver
        self.opti.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend
        try:
            return self.opti.solve(), Xi_sol, XiN_sol

        except:
            print(traceback.format_exc())
            print('------------------------------------INFEASIBILITY---------------------------------------------')
            self.infeasibilities.append(k0)

            raise(SystemExit)
            return self.opti.debug, Xi_sol, XiN_sol


    def solve_feasibility(self):
        '''Solve the feasibility problem'''

        # Define the objective to penalize the use of slack variables
        self.opti_feas.minimize(self.alpha * self.XiN_f + self.alpha2*sum([self.Xi_f[:, kk].T @ self.Xi_f[:, kk] for kk in range(self.N)]))

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

        # Set values to initial condition parameters
        self.opti_feas.set_value(self.X0_f, x0)
        self.opti.set_value(self.X0, x0)
        sol, Xi_sol, XiN_sol = self.solve_safety_filter(unom, x0, k0)
        return sol.value(self.X), sol.value(self.U), Xi_sol, XiN_sol

    def clear_events(self):
        '''Clear event logs'''
        self.events = []
        self.events_log = {}



class OneStepSafetyFilter(SafetyFilter):
    def __init__(self, x0, f, params, constraints, options=None):
        params['N'] = 1
        super().__init__(x0=x0, f=f, params=params, constraints=constraints, options=options)


    def setup_opt_variables(self):
        '''Set up the optimization variables and related terms'''

        # Define optimization in casadi for safety filter and feasibility problems
        self.opti_feas = Opti()
        self.opti = Opti()

        # State, slack, input variables for  feasibility problem
        self.X0_f = self.opti_feas.parameter(self.nx, 1)
        self.X_f = self.opti_feas.variable(self.nx, self.N + 1)
        self.XiN_f = self.opti_feas.variable()
        self.U_f = self.opti_feas.variable(self.nu, self.N)

        # State, input variables for safety filter
        self.X0 = self.opti.parameter(self.nx,1)
        self.X = self.opti.variable(self.nx, self.N + 1)
        self.U = self.opti.variable(self.nu, self.N)
        self.XiN = self.opti.parameter()


    def setup_constraints(self, k0=0):
        '''Set up constraints
         k0: initial time
         '''

        # Define differential equation/difference equation equality constraint for safety filter and feasibility problem
        self.dynamics_constraints(X=self.X_f, U=self.U_f, opt=self.opti_feas, k0=k0)
        self.dynamics_constraints(X=self.X, U=self.U, opt=self.opti, k0=k0)

        # Define state , input, constraints
        self.define_constraints(X=self.X_f, U=self.U_f, opt=self.opti_feas, x0=self.X0_f, k0=k0, XiN=self.XiN_f)
        self.define_constraints(X=self.X, U=self.U, opt=self.opti, x0=self.X0, k0=k0, XiN=self.XiN)

        # Define non-negativity constraints for slack terms of the feasibility problem
        self.opti_feas.subject_to(self.XiN_f >= 0.0)

        # Define casadi optimization parameters
        self.p_opts = {"expand": True, "print_time": False, "verbose": False}
        self.s_opts = { 'max_iter': 300,
                        'print_level': 1,
                        'warm_start_init_point': 'yes',
                        'tol': 1e-6,
                        'constr_viol_tol': 1e-6,
                        "compl_inf_tol": 1e-6,
                        "acceptable_tol": 1e-6,
                        "acceptable_constr_viol_tol": 1e-6,
                        "acceptable_dual_inf_tol": 1e-6,
                        "acceptable_compl_inf_tol": 1e-6,
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
            x_next = integrate(self.f, X[:, k], k0 + k, U[:, k], self.dT, self.integrator_type)
            opt.subject_to(X[:, k + 1] == x_next)


    def define_constraints(self, X, U, opt, x0, k0=0, XiN=None):
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
            XiN_sol = feas_sol.value(self.XiN_f)
            self.slacks_term.append( XiN_sol )

            # Save solutions for warm start
            self.Xf_warm = feas_sol.value(self.X_f)
            self.Uf_warm = feas_sol.value(self.U_f)
            self.XiN_warm = XiN_sol
            try:
                self.lamgf_warm = feas_sol.value(self.opti_feas.lam_g) # dual variables
            except:
                print('initialized no dual variables yet for feasibility problem')

        else:
            XiN_sol = 0.0

        # Set slack variables and control objective to remain minimally close to unom
        self.opti.set_value(self.XiN, XiN_sol)
        self.opti.minimize(dot(self.U[:, 0] - unom(x0, k0),self.U[:, 0] - unom(x0, k0) ))

        # Warm start the optimization
        self.opti.set_initial(self.X, self.Xf_warm)
        self.opti.set_initial(self.U, self.Uf_warm)
        if self.options['use_feas']:
            self.opti.set_initial(self.opti.lam_g, self.lamgf_warm[:-((self.N)*self.nc+1)]) # Removing lambdas associated with non-negative constraint in feasibility problem

        # Set the solver
        self.opti.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend
        try:
            return self.opti.solve(), XiN_sol
        except:
            print(traceback.format_exc())
            print('------------------------------------INFEASIBILITY---------------------------------------------')
            self.infeasibilities.append(k0)

            return self.opti.debug, XiN_sol


    def solve_feasibility(self):
        '''Solve the feasibility problem'''

        # Define the objective to penalize the use of slack variables
        self.opti_feas.minimize(self.alpha * self.XiN_f )

        # Warm start the optimization
        self.opti_feas.set_initial(self.X_f, self.Xf_warm)
        self.opti_feas.set_initial(self.U_f, self.Uf_warm)
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
        sol, XiN_sol = self.solve_safety_filter(unom, x0, k0)
        return sol.value(self.X), sol.value(self.U), XiN_sol




def integrate(f, x, k, u, dT, integrator_type='Euler'):
    '''Integrate system f at time k by one time step with x and u and time k using integrator_type method
    f: system dynamics function f(x,k,u)
    x: state
    k: time step
    u: control input
    dT: time step
    integrator_type: must be 'RK4', 'Euler', or 'cont'
    '''

    if integrator_type == 'RK4':
        # Runge-Kutta 4 integration
        k1 = f(x, k, u)
        k2 = f(x + dT / 2 * k1, k + dT / 2, u)
        k3 = f(x + dT / 2 * k2, k + dT / 2, u)
        k4 = f(x + dT * k3, k + dT, u)
        x_next = x + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    if integrator_type == 'Euler':
        # Standard Euler integration
        x_next = x + dT * f(x, k, u)

    if integrator_type == 'cont':
        # Treat system as 'continuous'
        x_next = f(x, k, u)

    return x_next

def integrate_adaptive(f, x, k, u, theta, dT, integrator_type='Euler'):
    '''Integrate system f at time k by one time step with x and u and time k using integrator_type method
    f: system dynamics function f(x,k,u, theta)
    x: state
    k: time step
    u: control input
    theta: model parameter
    dT: time step
    integrator_type: must be 'RK4', 'Euler', or 'cont'
    '''

    if integrator_type == 'RK4':
        # Runge-Kutta 4 integration
        k1 = f(x, k, u, theta)
        k2 = f(x + dT / 2 * k1, k + dT / 2, u, theta)
        k3 = f(x + dT / 2 * k2, k + dT / 2, u, theta)
        k4 = f(x + dT * k3, k + dT, u, theta)
        x_next = x + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    if integrator_type == 'Euler':
        # Standard Euler integration
        x_next = x + dT * f(x, k, u, theta)

    if integrator_type == 'cont':
        # Treat system as 'continuous'
        x_next = f(x, k, u, theta)

    return x_next