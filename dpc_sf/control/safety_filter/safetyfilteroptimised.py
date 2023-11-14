"""
written before the code has been started:

The goal of this code is to optimize the safety filter code to make
a fair comparison with MPC
"""

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

class SafetyFilter:
    def __init__(self, x0, f, params, constraints, options=None) -> None:

        # Initialize terms
        self.N = params['N']                                        # Prediction horizon
        self.delta = params['delta']                                # Constraint tightening parameter
        self.dts = params['dts']                                    # Sampling time
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

        # We instantiate both optimisers and the variables they will optimise as variables
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

        # define variable starting point constraint parameter for feasiblity and SF problems
        self.x0_f_para = self.opti_feas.parameter(self.nx, 1)
        self.x0_para = self.opti.parameter(self.nx, 1)

        # constraints of safety filter problem are a function of the outputs of the feasibility = parameters required
        # self.X_para = self.opti.parameter(self.nx, self.N + 1)
        self.Xi_para = self.opti.parameter(self.nc, self.N - 1)
        # self.XiN_para = self.opti.parameter()
        # self.U_para = self.opti.parameter(self.nu, self.N)
        self.unom0 = self.opti.parameter(self.nu, 1)

        # setup the dynamics constraints for opti
        self.dynamics_constraints(X=self.X, U=self.U, opt=self.opti, dts=self.dts)
        self.dynamics_constraints(X=self.X_f, U=self.U_f, opt=self.opti_feas, dts=self.dts)

        # Define state , input, constraints for the feasibility problem
        # self.define_constraints(X=self.X_f, U=self.U_f, opt=self.opti_feas, x0=x0, k0=0, Xi=self.Xi_f, XiN=self.XiN_f)

        # initial condition for feasibility
        self.opti_feas.set_value(self.x0_f_para, x0)
        self.opti_feas.subject_to(self.X_f[:,0] == self.x0_f_para)

        # State trajectory constraints, enforced at each time step
        for ii in range(self.nc):
            for kk in range(1,self.N):
                self.opti_feas.subject_to(self.state_constraints[ii](self.X_f[:, kk],kk) <= -kk*self.delta + self.Xi_f[ii,kk-1] - self.rob_marg)

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
    
        # Set up variables to warm start the optimization problems
        self.X_warm = np.zeros((self.nx, self.N+1))
        self.U_warm = np.zeros((self.nu, self.N))
        self.Xf_warm = np.zeros((self.nx, self.N+1))
        self.Uf_warm = np.zeros((self.nu, self.N))
        self.Xi_warm = np.zeros((self.nc, self.N-1))
        self.XiN_warm = 0.0
        
        # Solve feasiblity once
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

        # Define the solver and solve
        self.opti_feas.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend
        feas_sol = self.opti_feas.solve()

        # Save solutions for warm start
        Xi_sol = feas_sol.value(self.Xi_f)
        XiN_sol = feas_sol.value(self.XiN_f)

        # repeat final solution, ignore first for next timestep
        self.Xf_warm = np.hstack([feas_sol.value(self.X_f)[:,1:], feas_sol.value(self.X_f)[:,-1:]])
        self.Uf_warm = np.hstack([feas_sol.value(self.U_f)[:,1:], feas_sol.value(self.U_f)[:,-1:]])
        self.Xi_warm = np.hstack([Xi_sol[:,1:], Xi_sol[:,-1:]])
        self.XiN_warm = XiN_sol

        self.lamgf_warm = feas_sol.value(self.opti_feas.lam_g) # dual variables

        # constraints for the SF based on outputs of the feasibility problem

        # initial condition for feasibility
        self.opti.set_value(self.x0_para, x0)
        self.opti.subject_to(self.X[:,0] == self.x0_para)

        # State trajectory constraints, enforced at each time step
        self.opti.set_value(self.Xi_para, self.Xi_warm)
        for ii in range(self.nc):
            for kk in range(1,self.N):
                self.opti.subject_to(self.state_constraints[ii](self.X[:, kk],kk) <= -kk*self.delta + self.Xi_para[ii,kk-1] - self.rob_marg)

        # Define constraints for the feasibility problem and control objective to remain minimally close to unom
        unom0_dummy = np.array([[0.,0.,0.,0.]]).T
        self.opti.set_value(self.unom0, unom0_dummy)
        self.opti.minimize(dot(self.U[:, 0] - self.unom0, self.U[:, 0] - self.unom0)) #self.opti.minimize((self.U[:, 0] - unom(x0, k0)).T @ (self.U[:, 0] - unom(x0,k0)))

        # Warm start the optimization
        self.opti.set_initial(self.X, self.Xf_warm)
        self.opti.set_initial(self.U, self.Uf_warm)
        self.opti.set_initial(self.opti.lam_g, self.lamgf_warm[:-((self.N-1)*self.nc+1)]) # Removing lambdas associated with non-negative constraint in feasibility problem

        # Set the solver
        self.opti.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend

        try:
            sf_sol = self.opti.solve()
            
            # save variables for warm starting next iteration
            X_sol = sf_sol.value(self.X)
            U_sol = sf_sol.value(self.U)

            self.X_warm = np.hstack([X_sol[:,1:], X_sol[:,-1:]])
            self.U_warm = np.hstack([U_sol[:,1:], U_sol[:,-1:]])

        except:
            print('------------------------------------INFEASIBILITY---------------------------------------------')

        # try:
        #     return self.opti.solve(), Xi_sol, XiN_sol
        # except:
        #     print(traceback.format_exc())
        #     print('------------------------------------INFEASIBILITY---------------------------------------------')
        #     self.infeasibilities.append(k0)
        #     return self.opti.debug, Xi_sol, XiN_sol

        print('fin')

    def __call__(self, state, unom):

        # Feasibility Problem Solve
        # -------------------------

        # update feasibility parameters
        self.opti_feas.set_value(self.x0_f_para, state)

        # Warm start the optimization
        self.opti_feas.set_initial(self.X_f, self.Xf_warm)
        self.opti_feas.set_initial(self.U_f, self.Uf_warm)
        self.opti_feas.set_initial(self.Xi_f, self.Xi_warm)
        self.opti_feas.set_initial(self.XiN_f, self.XiN_warm)

        # solve feasibility problem
        feas_sol = self.opti_feas.solve()

        # Save solutions for warm start
        Xi_sol = feas_sol.value(self.Xi_f)
        XiN_sol = feas_sol.value(self.XiN_f)

        # repeat final solution, ignore first for next timestep
        self.Xf_warm = np.hstack([feas_sol.value(self.X_f)[:,1:], feas_sol.value(self.X_f)[:,-1:]])
        self.Uf_warm = np.hstack([feas_sol.value(self.U_f)[:,1:], feas_sol.value(self.U_f)[:,-1:]])
        self.Xi_warm = np.hstack([Xi_sol[:,1:], Xi_sol[:,-1:]])
        self.XiN_warm = XiN_sol
        self.lamgf_warm = feas_sol.value(self.opti_feas.lam_g) # dual variables

        # Safety Filter Solve
        # -------------------

        # update safety filter parameters
        self.opti.set_value(self.x0_para, state)
        self.opti.set_value(self.Xi_para, self.Xi_warm)
        self.opti.set_value(self.unom0, unom)

        # Warm start the optimization
        self.opti.set_initial(self.X, self.Xf_warm)
        self.opti.set_initial(self.U, self.Uf_warm)
        self.opti.set_initial(self.opti.lam_g, self.lamgf_warm[:-((self.N-1)*self.nc+1)]) # Removing lambdas associated with non-negative constraint in feasibility problem
        
        try:
            sf_sol = self.opti.solve()
            
            # save variables for warm starting next iteration
            X_sol = sf_sol.value(self.X)
            U_sol = sf_sol.value(self.U)

            self.X_warm = np.hstack([X_sol[:,1:], X_sol[:,-1:]])
            self.U_warm = np.hstack([U_sol[:,1:], U_sol[:,-1:]])

            self.last_feas_u = U_sol[:,0]
            return U_sol[:,0]

        except:
            print('------------------------------------INFEASIBILITY---------------------------------------------')
            return self.last_feas_u
        
    def dynamics_constraints(self, X, U, opt, dts, k0=0):
        '''Defines the constraints related to the ode of the system dynamics
        X: system state
        U: system input
        opt: Casadi optimization class'''
        if self.integrator_type == 'RK4':
            # Runge-Kutta 4 integration
            for k in range(self.N):  # loop over control intervals (these are the equality constraints associated with the dynamics)
                # current_time = sum1(dts[:k])
                k1 = self.f(X[:, k], 0, U[:, k])
                k2 = self.f(X[:, k] + dts[k] / 2 * k1, 0, U[:, k])
                k3 = self.f(X[:, k] + dts[k] / 2 * k2, 0, U[:, k])
                k4 = self.f(X[:, k] + dts[k] * k3, 0, U[:, k])
                x_next = X[:, k] + dts[k] / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                opt.subject_to(X[:, k + 1] == x_next)  # close the gaps

        if self.integrator_type == 'Euler':
            # Standard Euler integration
            for k in range(self.N):  # loop over control intervals (these are the equality constraints associated with the dynamics)
                x_next = X[:, k] + dts[k] * self.f(X[:, k], 0, U[:, k])
                opt.subject_to(X[:, k + 1] == x_next)  # close the gaps

        if self.integrator_type == 'cont':
            # Treat system as 'continuous'
            for k in range(self.N):  # loop over control intervals (these are the equality constraints associated with the dynamics)
                x_next = self.f(X[:, k], 0, U[:, k])
                opt.subject_to(X[:, k + 1] == x_next)  # close the gaps

if __name__ == "__main__":
    # the code below has been combined with the old safety filter code to have a better test structure :)
    import dpc_sf.control.safety_filter.unit_test_opt_sf
#     ### basic parameters
#     nx = 17
#     nu = 4
#     N = 30     # prediction horizon No. steps
#     Nsim = 300  # simulation length
# 
#     Ts = 0.01
#     Tf_hzn = 3.0
#     N = 300
#     Ti = 0.0
#     Tf = 15.0
#     integrator_type = "Euler"
#     obstacle_opts = {'r': 0.5, 'x': 1, 'y': 1}
#     # Find the optimal dts for the MPC
#     dt_1 = Ts
#     d = (2 * (Tf_hzn/N) - 2 * dt_1) / (N - 1)
#     dts = [dt_1 + i * d for i in range(N)]
#     ### dummy policy
#     policy = lambda x, t: np.array([0.,0.,0.,0.])
#     ### terminal constraint
#     xr = np.copy(quad_params["default_init_state_np"])
#     xr[0] = 2 # x  
#     xr[1] = 2 # y
#     xr[2] = 1 # z flipped
#     ### create dynamics
#     quad = QuadcopterCA()
#     def f(x, t, u):
#         return quad.state_dot_1d(x, u)[:,0]
#     ### define SF parameters
#     params = {}
#     params['N'] = N  # prediction horizon
#     params['Ts'] = Ts  # sampling time
#     params['dts'] = dts
#     params['delta'] = 0.00005  # Small robustness margin for constraint tightening
#     params['robust_margin'] = 0.1 #0.0 #  #   Robustness margin for handling perturbations
#     params['robust_margin_terminal'] = 0.0001 #0.0 # # Robustness margin for handling perturbations (terminal set)
#     params['alpha2'] = 1.0  # Multiplication factor for feasibility parameters
#     params['integrator_type'] = integrator_type  # Integrator type for the safety filter
#     params['nx'] = nx
#     params['nu'] = nu
#     ### System constraints
#     terminal_constraint = {}
#     state_constraints = {}
#     input_constraints = {}
#     # state upper/lower bounds
#     umax = [quad.params["maxCmd"]]  #model.umax*np.ones((nu,1)) # Input constraint parameters(max)
#     umin = [quad.params["minCmd"]] #model.umin*np.ones((nu,1)) # Input constraint parameters (min)
#     scu_k = quad.params["state_ub"] 
#     scl_k = quad.params["state_lb"]
#     scu_k[0:13] *= 100
#     scl_k[0:13] *= 100
#     state_constraints[0] = lambda x, k: x - scu_k
#     state_constraints[1] = lambda x, k: scl_k - x
#     # cylinder constraint
#     x_c = 1.0
#     y_c = 1.0
#     r = 0.51
#     def cylinder_constraint(x, k):
#         current_time = sum1(dts[:k])
#         multiplier = 1 + current_time * 0.1
#         current_x, current_y = x[0], x[1]
#         return r ** 2 * multiplier - (current_x - x_c)**2 - (current_y - y_c)**2
#     state_constraints[2] = cylinder_constraint# lambda x, k: r**2 * (1 + k * dt * 0.001) - (x[0] - x_c)**2 - (x[1] - y_c)**2
#     # Quadratic CBF (assuming linearisation valid)
#     cbf_const = (np.sqrt(2) - 0.5) ** 2
#     hf = lambda x: (x-xr).T @ (x-xr) - cbf_const
#     # hf = lambda x: r**2 - (x[0] - x_c)**2 - (x[1] - y_c)**2
#     # input constraints
#     kk = 0
#     nu = 4
#     for ii in range(nu):
#         input_constraints[kk] = lambda u: u[ii] - umax[0]
#         kk += 1
#         input_constraints[kk] = lambda u: umin[0] - u[ii]
#         kk += 1    
#     # combine constraints to send to SF
#     constraints = {}
#     constraints['state_constraints'] = state_constraints
#     constraints['hf'] = hf
#     constraints['input_constraints'] = input_constraints    
#     ### instantiate safety filter
#     x0 = quad_params["default_init_state_np"]
#     SF = SafetyFilter(x0=x0, f=f, params=params, constraints=constraints, options={})
#     # CL = sf.ClosedLoopSystem(f=f_pert, u=self.SF, u_nom=self.unom, dt=dt, int_type='Euler', use_true_unom=True)
#     u = SF(x0, np.array([10.,0.,-10.,0.]))
#     print('fin')