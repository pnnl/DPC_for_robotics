import traceback
import numpy as np
import torch
import neuromancer as nm
from neuromancer.dynamics import integrators
import casadi as ca
import copy
from tqdm import tqdm
import time
from utils.time import time_function

import utils.pytorch as ptu
from dpc import posVel2cyl
from utils.integrate import euler, RK4, generate_variable_timesteps, generate_variable_times
from dynamics import get_quad_params, mujoco_quad, state_dot
from proportional import get_ctrl_params, PID
import reference
from utils.quad import Animator
from utils.quad import Animator, plot_mujoco_trajectories_wp_p2p, calculate_mpc_cost, plot_mujoco_trajectories_wp_traj
from barrier_function import BarrierFunction
from scipy.spatial import distance

class SafetyFilterOld:

    """
    The time-invariant variable-timestep robust predictive safety filter implementation
    """

    def __init__(self, dynamics, Ts, Tf_hzn, N, quad_params, integrator, bf=None):

        """ 
        dynamics    - the quad dynamics themselves in casadi
        Ts          - timestep
        Tf_hzn      - the predictive horizon final time
        N           - the number of steps in the predictive horizon (variable in length according to generate_variable_timesteps)
        quad_params - the literal quad parameters
        """

        self.dynamics = dynamics
        dts = generate_variable_timesteps(Ts, Tf_hzn, N)
        self.quad_params = quad_params
        self.integrator = integrator
        self.N = N

        cylinder_expansion_multiplier = 1.0 # 0.001

        self.nx = 17
        self.nu = 4
        delta = 0.00005 # Small robustness margin for constraint tightening
        # delta = 0.0001
        # delta = 0.01
        # delta = 0.0
        # alpha = 10000.0  # Gain term required to be large as per Wabersich2022 paper
        alpha = 1000000.0
        alpha2 = 1.0 # Multiplication factor for feasibility parameters
        Lh_x, Ld, Lf_x, Lhf_x = 1.0, 0.01, 1.0, 1.0 # 0., 0., 0., 0.

        # Define Box Constraints
        # -------------------------
        state_constraints = {}
        input_constraints = {}

        umax = [quad_params["maxCmd"]*1] # *100  #model.umax*np.ones((nu,1)) # Input constraint parameters(max)
        umin = [quad_params["minCmd"]*1] # *100 #model.umin*np.ones((nu,1)) # Input constraint parameters (min)

        scu_k = quad_params["state_ub"] 
        scl_k = quad_params["state_lb"]
        scu_k[0:13] *= 100
        scl_k[0:13] *= 100
        
        # scu_k[7], scu_k[8] = 1.0, 1.0       # mimicing the convex hull
        # scl_k[7], scl_k[8] = -1.0, -1.0     # mimicing the convex hull
        state_constraints[0] = lambda x, k: x - scu_k
        state_constraints[1] = lambda x, k: scl_k - x
        # cylinder constraint
        x_c, y_c, r = 1.0, 1.0, 0.50 # 0.51
        # state_constraints[2] = lambda x, k: r**2 * (1 + k * dts[k] * 0.001) - (x[0] - x_c)**2 - (x[1] - y_c)**2
        def cyl_constraint(x, k):
            print(x[0].shape)
            print(x[1].shape)
            return r**2 * (1 + k * dts[k] * cylinder_expansion_multiplier) - (x[0] - x_c)**2 - (x[1] - y_c)**2
        state_constraints[2] = cyl_constraint
            
        # Input constraints: h(u) <= 0
        kk = 0
        for ii in range(self.nu):
            input_constraints[kk] = lambda u: u[ii] - umax[0]
            kk += 1
            input_constraints[kk] = lambda u: umin[0] - u[ii]
            kk += 1   

        self.nc = len(state_constraints) # number of state constraints
        self.ncu = len(input_constraints) # number of input constraints

        # Define CasADi optimisation objects
        # ----------------------------------
        self.opti_feas = ca.Opti()
        self.opti = ca.Opti()

        # State, slack, input variables for  feasibility problem
        self.X0_f   = self.opti_feas.parameter(self.nx, 1)
        self.X_f    = self.opti_feas.variable(self.nx, N + 1)
        self.Xi_f   = self.opti_feas.variable(self.nc, N)
        self.XiN_f  = self.opti_feas.variable()
        self.U_f    = self.opti_feas.variable(self.nu, N)

        # State, input variables for safety filter
        self.X0     = self.opti.parameter(self.nx, 1)
        self.X      = self.opti.variable(self.nx, N + 1)
        self.U      = self.opti.variable(self.nu, N)
        self.Xi     = self.opti.parameter(self.nc, N)
        self.XiN    = self.opti.parameter()
        self.Unom   = self.opti.parameter(self.nu, 1)

        # Set up variables to warm start the optimization problems
        self.X_warm = np.zeros((self.nx, N+1))
        self.U_warm = np.zeros((self.nu, N))
        self.Xf_warm = np.zeros((self.nx, N + 1))
        self.Uf_warm = np.zeros((self.nu, N))
        self.Xi_warm = np.zeros((self.nc, N))
        self.XiN_warm = 0.0

        # Define casadi optimization parameters
        self.p_opts = {"expand": True, "print_time": False, "verbose": False}
        self.s_opts = {"max_iter": 5000, "print_level": 1, "tol": 1e-6}
        # self.s_opts = { 'max_iter': 300,
        #                 'print_level': 1,
        #                 # 'warm_start_init_point': 'yes',
        #                 'tol': 1e-8,
        #                 'constr_viol_tol': 1e-8,
        #                 "compl_inf_tol": 1e-8,
        #                 "acceptable_tol": 1e-4,
        #                 "acceptable_constr_viol_tol": 1e-8,
        #                 "acceptable_dual_inf_tol": 1e-8,
        #                 "acceptable_compl_inf_tol": 1e-8,
        #                 }

        # Define System Constraints
        # -------------------------

        # dynamics constraints
        for k in range(self.N):
            self.opti_feas.subject_to(self.X_f[:, k + 1] == self.integrator(self.dynamics, self.X_f[:, k], self.U_f[:, k], dts[k], quad_params))
            self.opti.subject_to(self.X[:, k + 1] == self.integrator(self.dynamics, self.X[:, k], self.U[:, k], dts[k], quad_params))
        
        # Trajectory constraints,
        for ii in range(self.nx):
            # initial condition constraint
            self.opti_feas.subject_to(self.X_f[ii, 0] == self.X0_f[ii])
            self.opti.subject_to(self.X[ii, 0] == self.X0[ii])

        # State trajectory constraints, enforced at each time step
        for ii in range(self.nc):
            for kk in range(0, self.N):
                # compute robustness margin:
                if kk > 0.0:
                    rob_marg = kk * delta + Lh_x * Ld * sum([Lf_x**j for j in range(kk)])
                else:
                    rob_marg = 0.0
                self.opti_feas.subject_to(state_constraints[ii](self.X_f[:, kk],kk) <= self.Xi_f[ii,kk] - rob_marg)
                self.opti.subject_to(state_constraints[ii](self.X[:, kk],kk) <= self.Xi[ii,kk] - rob_marg)   

        # Input constraints
        for iii in range(self.ncu):
            for kkk in range(self.N):
                self.opti_feas.subject_to(input_constraints[iii](self.U_f[:, kkk]) <= 0.0)
                self.opti.subject_to(input_constraints[iii](self.U[:, kkk]) <= 0.0)

        # Define non-negativity constraints for slack terms of the feasibility problem
        [self.opti_feas.subject_to(self.Xi_f[ii,:] >= 0.0) for ii in range(self.nc)]
        self.opti_feas.subject_to(self.XiN_f >= 0.0)

        # Define the cost functions
        # -------------------------

        # Define the feasibility problem objective to penalize the use of slack variables
        self.opti_feas.minimize(alpha * self.XiN_f + alpha2*sum([self.Xi_f[:, kk].T @ self.Xi_f[:, kk] for kk in range(N)]))
        # Safety Filter
        self.opti.minimize(ca.dot(self.U[:,0] - self.Unom[:,0], self.U[:,0] - self.Unom[:,0]))

        # Warm start the optimization
        self.opti_feas.set_initial(self.X_f, self.Xf_warm)
        self.opti_feas.set_initial(self.U_f, self.Uf_warm)
        self.opti_feas.set_initial(self.Xi_f, self.Xi_warm)
        self.opti_feas.set_initial(self.XiN_f, self.XiN_warm)
        try:
            self.opti_feas.set_initial(self.opti_feas.lam_g, self.lamgf_warm)
        except:
            print('initialized no dual variables yet for feasibility problem')

        # Define the solvers
        self.opti_feas.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend
        self.opti.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend

    def __call__(self, x_seq, u_seq, violation):

        print(f"dist2cyl: {(x_seq.flatten()[0] - 1)**2 + (x_seq.flatten()[1] - 1)**2 - 0.5**2}")

        u_seq = ptu.to_numpy(u_seq.T)
        x_seq = ptu.to_numpy(x_seq.T)

        if not violation:

            # Save solution for warm start
            # self.X_warm = x_seq
            # self.U_warm = u_seq
            # self.Xf_warm = x_seq
            # self.Uf_warm = u_seq
            # self.Xi_warm = np.zeros((self.nc, self.N))
            # self.XiN_warm = 0.0
            return ptu.from_numpy(u_seq[:,0]), ptu.from_numpy(self.X_warm)
        
        else:

            # solve the feasibility problem:
            # ------------------------------

            # Set values to initial condition parameters, set initial values to optimisation parameters
            self.opti_feas.set_value(self.X0_f, x_seq[:,0])
            self.opti_feas.set_initial(self.X_f, self.Xf_warm)
            self.opti_feas.set_initial(self.U_f, self.Uf_warm)
            self.opti_feas.set_initial(self.Xi_f, self.Xi_warm)
            self.opti_feas.set_initial(self.XiN_f, self.XiN_warm)
            try:
                self.opti_feas.set_initial(self.opti_feas.lam_g, self.lamgf_warm)
            except:
                print('initialized no dual variables yet for feasibility problem')

            feas_sol = self.opti_feas.solve()

            # Save solutions for warm start
            self.Xf_warm = feas_sol.value(self.X_f)
            self.Uf_warm = feas_sol.value(self.U_f)
            self.Xi_warm = feas_sol.value(self.Xi_f)
            self.XiN_warm = feas_sol.value(self.XiN_f)
            self.lamgf_warm = feas_sol.value(self.opti_feas.lam_g) # dual variables

            print(f"Xi_sol : {feas_sol.value(self.Xi_f)[0,0]}")
            print(f"XiN_sol : {feas_sol.value(self.XiN_f)}")

            # solve the safety filter:
            # ------------------------

            # Set values to initial condition parameters, slack variables and control objective to remain minimally close to unom
            self.opti.set_value(self.X0, x_seq[:,0])
            self.opti.set_value(self.Xi, feas_sol.value(self.Xi_f))
            self.opti.set_value(self.XiN, feas_sol.value(self.XiN_f))
            self.opti.set_value(self.Unom, u_seq[:,0])

            # Warm start the optimization
            self.opti.set_initial(self.X, self.Xf_warm)
            self.opti.set_initial(self.U, self.Uf_warm)
            self.opti.set_initial(self.opti.lam_g, self.lamgf_warm[:-((self.N)*(self.nc)+1)]) # Removing lambdas associated with non-negative constraint in feasibility problem

            try:
                sol = self.opti.solve()
            except:
                print(traceback.format_exc())
                print('------------------------------------INFEASIBILITY---------------------------------------------')
                print('using nominal DPC control...')
                return ptu.from_numpy(u_seq[:,0]) # self.opti.debug, feas_sol.value(self.Xi_f), feas_sol.value(self.XiN_f)
            
            # Save solution for warm start
            self.X_warm = sol.value(self.X)
            self.U_warm = sol.value(self.U)
            self.lamg_warm = sol.value(self.opti.lam_g)

            u = ptu.from_numpy(np.array(sol.value(self.U[:,0])).reshape((self.nu,1))[:,0])

            print(f"u delta: {ptu.from_numpy(u_seq[:,0]) - u}")

            return u, ptu.from_numpy(self.X_warm)

class SafetyFilter:

    """
    The time-invariant variable-timestep robust predictive safety filter implementation
    """

    def __init__(self, dynamics, Ts, Tf_hzn, N, quad_params, integrator, bf=None):

        """ 
        dynamics    - the quad dynamics themselves in casadi
        Ts          - timestep
        Tf_hzn      - the predictive horizon final time
        N           - the number of steps in the predictive horizon (variable in length according to generate_variable_timesteps)
        quad_params - the literal quad parameters
        """

        self.dynamics = dynamics
        dts = generate_variable_timesteps(Ts, Tf_hzn, N)
        self.quad_params = quad_params
        self.integrator = integrator
        self.N = N
        self.bf = bf

        self.nx = 17
        self.nu = 4
        self.nvc = 2
        delta = 0.00005 # Small robustness margin for constraint tightening
        alpha = 1000000.0
        alpha2 = 1.0 # Multiplication factor for feasibility parameters
        Lh_x, Ld, Lf_x, Lhf_x = 0, 0, 0, 0

        # Define Box Constraints
        # -------------------------
        state_constraints = {}
        input_constraints = {}

        umax = [quad_params["maxCmd"]*1] # *100  #model.umax*np.ones((nu,1)) # Input constraint parameters(max)
        umin = [quad_params["minCmd"]*1] # *100 #model.umin*np.ones((nu,1)) # Input constraint parameters (min)

        scu_k = quad_params["state_ub"] 
        scl_k = quad_params["state_lb"]
        scu_k[0:13] *= 100
        scl_k[0:13] *= 100
        
        state_constraints[0] = lambda x, k: x - scu_k
        state_constraints[1] = lambda x, k: scl_k - x

        # cylinder parameters
        x_c, y_c, r = 1.0, 1.0, 0.50 # 0.51

        # Input constraints: h(u) <= 0
        kk = 0
        for ii in range(self.nu):
            input_constraints[kk] = lambda u: u[ii] - umax[0]
            kk += 1
            input_constraints[kk] = lambda u: umin[0] - u[ii]
            kk += 1   

        self.nc = len(state_constraints) # number of state constraints
        self.ncu = len(input_constraints) # number of input constraints

        # Define CasADi optimisation objects
        # ----------------------------------
        self.opti_feas = ca.Opti()
        self.opti = ca.Opti()

        # State, slack, input variables for  feasibility problem
        self.X0_f   = self.opti_feas.parameter(self.nx, 1)
        self.X_f    = self.opti_feas.variable(self.nx, N + 1)
        self.Xi_f   = self.opti_feas.variable(self.nc + self.nvc, N) # nc + 1 for hyperplane constraint
        self.XiN_f  = self.opti_feas.variable()
        self.U_f    = self.opti_feas.variable(self.nu, N)
        self.a1_f    = self.opti_feas.parameter(6, 1)    # normal vector for hyperplane constraints
        self.b1_f    = self.opti_feas.parameter(6, 1)    # offset for hyperplane constraints
        self.a2_f    = self.opti_feas.parameter(2, 1)    # normal vector for hyperplane constraints
        self.b2_f    = self.opti_feas.parameter(2, 1)    # offset for hyperplane constraints

        # State, input variables for safety filter
        self.X0     = self.opti.parameter(self.nx, 1)
        self.X      = self.opti.variable(self.nx, N + 1)
        self.U      = self.opti.variable(self.nu, N)
        self.Xi     = self.opti.parameter(self.nc + self.nvc, N) # nc + 1 for hyperplane constraint
        self.XiN    = self.opti.parameter()
        self.Unom   = self.opti.parameter(self.nu, 1)
        self.a1      = self.opti.parameter(6, 1)         # normal vector for hyperplane constraints
        self.b1      = self.opti.parameter(6, 1)         # offset for hyperplane constraints
        self.a2      = self.opti.parameter(2, 1)         # normal vector for hyperplane constraints
        self.b2      = self.opti.parameter(2, 1)         # offset for hyperplane constraints

        # Set up variables to warm start the optimization problems
        self.X_warm = np.zeros((self.nx, N+1))
        self.U_warm = np.zeros((self.nu, N))
        self.Xf_warm = np.zeros((self.nx, N + 1))
        self.Uf_warm = np.zeros((self.nu, N))
        self.Xi_warm = np.zeros((self.nc + self.nvc, N))
        self.XiN_warm = 0.0

        # Define casadi optimization parameters
        self.p_opts = {"expand": True, "print_time": False, "verbose": False}
        self.s_opts = {"max_iter": 5000, "print_level": 1, "tol": 1e-6}

        # Define System Constraints
        # -------------------------

        # dynamics constraints
        for k in range(self.N):
            self.opti_feas.subject_to(self.X_f[:, k + 1] == self.integrator(self.dynamics, self.X_f[:, k], self.U_f[:, k], dts[k], quad_params))
            self.opti.subject_to(self.X[:, k + 1] == self.integrator(self.dynamics, self.X[:, k], self.U[:, k], dts[k], quad_params))
        
        # Trajectory constraints,
        for ii in range(self.nx):
            # initial condition constraint
            self.opti_feas.subject_to(self.X_f[ii, 0] == self.X0_f[ii])
            self.opti.subject_to(self.X[ii, 0] == self.X0[ii])

        # State trajectory constraints, enforced at each time step
        for ii in range(self.nc):
            for kk in range(0, self.N):
                # compute robustness margin:
                if kk > 0.0:
                    rob_marg = kk * delta + Lh_x * Ld * sum([Lf_x**j for j in range(kk)])
                else:
                    rob_marg = 0.0
                self.opti_feas.subject_to(state_constraints[ii](self.X_f[:, kk],kk) <= self.Xi_f[ii,kk] - rob_marg)
                self.opti.subject_to(state_constraints[ii](self.X[:, kk],kk) <= self.Xi[ii,kk] - rob_marg)

        # Safe Set trajectory constraints, enforced at each time step
        self.dpc_obs_to_pv = lambda obs: posVel2cyl.casadi(obs, np.array([[x_c,y_c]]).T, r)

        for ii in range(self.nvc):
            if ii == 0:

                for kk in range(self.N):
                    if kk > 0.0:
                        rob_marg = kk * delta + Lh_x * Ld * sum([Lf_x**j for j in range(kk)])
                    else:
                        rob_marg = 0.0
                    obs_f = ca.vertcat(*[self.X_f[i, kk] for i in [0,7,1,8,2,9]])
                    obs = ca.vertcat(*[self.X[i, kk] for i in [0,7,1,8,2,9]])
                    self.opti_feas.subject_to(self.a1_f.T @ obs_f + self.b1_f <= self.Xi_f[self.nc + ii, kk] - rob_marg)
                    self.opti.subject_to(self.a1.T @ obs + self.b1 <= self.Xi[self.nc + ii, kk] - rob_marg)         

            elif ii == 1: # the pv constraints
                for kk in range(self.N):
                    if kk > 0.0:
                        rob_marg = kk * delta + Lh_x * Ld * sum([Lf_x**j for j in range(kk)])
                    else:
                        rob_marg = 0.0
                    obs_f = self.dpc_obs_to_pv(ca.vertcat(*[self.X_f[i, kk] for i in [0,7,1,8,2,9]]))
                    obs = self.dpc_obs_to_pv(ca.vertcat(*[self.X[i, kk] for i in [0,7,1,8,2,9]]))
                    obs_f[0] -= self.bf.cylinder_margin
                    obs[0] -= self.bf.cylinder_margin
                    self.opti_feas.subject_to(self.a2_f.T @ obs_f + self.b2_f <= self.Xi_f[self.nc + ii, kk] - rob_marg)
                    self.opti.subject_to(self.a2.T @ obs + self.b2 <= self.Xi[self.nc + ii, kk] - rob_marg)         

        # DPC training dataset safe set terminal constraint
        # xr = np.copy(quad_params["default_init_state_np"])

        # cbf_const = (np.sqrt(2) - 0.5) ** 2
        # terminal constraint set
        # xr = np.copy(quad_params["default_init_state_np"])
        # xr[0], xr[1], xr[2] = 2, 2, 1 # x, y, z, (z is flipped)
        # hf = lambda x: (x-xr).T @ (x-xr) - cbf_const

        # Terminal constraint, enforced at last time step
        # rob_marg_term = Lhf_x * Ld * Lf_x ** (self.N - 1)
        # self.opti_feas.subject_to(hf(self.X_f[:, -1]) <= self.XiN_f - rob_marg_term)
        # self.opti.subject_to(hf(self.X[:, -1]) <= self.XiN - rob_marg_term)

        # Input constraints
        for iii in range(self.ncu):
            for kkk in range(self.N):
                self.opti_feas.subject_to(input_constraints[iii](self.U_f[:, kkk]) <= 0.0)
                self.opti.subject_to(input_constraints[iii](self.U[:, kkk]) <= 0.0)

        # Define non-negativity constraints for slack terms of the feasibility problem
        [self.opti_feas.subject_to(self.Xi_f[ii,:] >= 0.0) for ii in range(self.nc + self.nvc)]
        self.opti_feas.subject_to(self.XiN_f >= 0.0)

        # Define the cost functions
        # -------------------------

        # Define the feasibility problem objective to penalize the use of slack variables
        self.opti_feas.minimize(alpha * self.XiN_f + alpha2*sum([self.Xi_f[:, kk].T @ self.Xi_f[:, kk] for kk in range(N)]))
        # Safety Filter
        self.opti.minimize(ca.dot(self.U[:,0] - self.Unom[:,0], self.U[:,0] - self.Unom[:,0]))

        # Warm start the optimization
        self.opti_feas.set_initial(self.X_f, self.Xf_warm)
        self.opti_feas.set_initial(self.U_f, self.Uf_warm)
        self.opti_feas.set_initial(self.Xi_f, self.Xi_warm)
        self.opti_feas.set_initial(self.XiN_f, self.XiN_warm)
        try:
            self.opti_feas.set_initial(self.opti_feas.lam_g, self.lamgf_warm)
        except:
            print('initialized no dual variables yet for feasibility problem')

        # Define the solvers
        self.opti_feas.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend
        self.opti.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend

    def __call__(self, x, u, violation):

        print(f"dist2cyl: {(x.flatten()[0] - 1)**2 + (x.flatten()[1] - 1)**2 - 0.5**2}")
        print(x)

        u = ptu.to_numpy(u.T)
        x = ptu.to_numpy(x.T)

        if not violation:

            # Save solution for warm start
            # self.X_warm = x_seq
            # self.U_warm = u_seq
            # self.Xf_warm = x_seq
            # self.Uf_warm = u_seq
            # self.Xi_warm = np.zeros((self.nc, self.N))
            # self.XiN_warm = 0.0
            return ptu.from_numpy(u[:,0]), ptu.from_numpy(self.X_warm)
        
        else:

            # solve the feasibility problem:
            # ------------------------------
            if self.bf is not None:
                # the convex hull
                obs = ca.vertcat(*[x[i, :] for i in [0,7,1,8,2,9]])
                normal_vec, offset, _ = self.bf.find_tangential_hyperplane_on_cvx_hull_nearest_point(obs)
                self.opti_feas.set_value(self.a1_f, normal_vec)
                self.opti_feas.set_value(self.b1_f, offset)
                self.opti.set_value(self.a1, normal_vec)
                self.opti.set_value(self.b1, offset)

                # the pv convex hull
                pv = self.dpc_obs_to_pv(obs)
                normal_vec_pv, offset_pv, _ = self.bf.find_tangential_hyperplane_on_pv_hull_nearest_point(pv)
                self.opti_feas.set_value(self.a2_f, normal_vec_pv)
                self.opti_feas.set_value(self.b2_f, offset_pv)
                self.opti.set_value(self.a2, normal_vec_pv)
                self.opti.set_value(self.b2, offset_pv)

            # Set values to initial condition parameters, set initial values to optimisation parameters
            if x[0,0] == 1. and x[1,0] == 1.:
                x[0:2,0] += 1e-3 # avoids being exactly in center of cylinder - confuses cylindrical coordinates
            self.opti_feas.set_value(self.X0_f, x[:,0])
            self.opti_feas.set_initial(self.X_f, self.Xf_warm)
            self.opti_feas.set_initial(self.U_f, self.Uf_warm)
            self.opti_feas.set_initial(self.Xi_f, self.Xi_warm)
            self.opti_feas.set_initial(self.XiN_f, self.XiN_warm)
            try:
                self.opti_feas.set_initial(self.opti_feas.lam_g, self.lamgf_warm)
            except:
                print('initialized no dual variables yet for feasibility problem')

            feas_sol = self.opti_feas.solve()

            # Save solutions for warm start
            self.Xf_warm = feas_sol.value(self.X_f)
            self.Uf_warm = feas_sol.value(self.U_f)
            self.Xi_warm = feas_sol.value(self.Xi_f)
            self.XiN_warm = feas_sol.value(self.XiN_f)
            self.lamgf_warm = feas_sol.value(self.opti_feas.lam_g) # dual variables

            print(f"Xi_sol : {feas_sol.value(self.Xi_f)[0,0]}")
            print(f"XiN_sol : {feas_sol.value(self.XiN_f)}")

            # solve the safety filter:
            # ------------------------

            # Set values to initial condition parameters, slack variables and control objective to remain minimally close to unom
            self.opti.set_value(self.X0, x[:,0])
            self.opti.set_value(self.Xi, feas_sol.value(self.Xi_f))
            self.opti.set_value(self.XiN, feas_sol.value(self.XiN_f))
            self.opti.set_value(self.Unom, u[:,0])

            # Warm start the optimization
            self.opti.set_initial(self.X, self.Xf_warm)
            self.opti.set_initial(self.U, self.Uf_warm)
            self.opti.set_initial(self.opti.lam_g, self.lamgf_warm[:-((self.N)*(self.nc + self.nvc)+1)]) # Removing lambdas associated with non-negative constraint in feasibility problem

            try:
                sol = self.opti.solve()
            except:
                print(traceback.format_exc())
                print('------------------------------------INFEASIBILITY---------------------------------------------')
                print('using nominal DPC control...')
                return ptu.from_numpy(u[:,0]), ptu.from_numpy(self.X_warm) # self.opti.debug, feas_sol.value(self.Xi_f), feas_sol.value(self.XiN_f)
            
            # Save solution for warm start
            self.X_warm = sol.value(self.X)
            self.U_warm = sol.value(self.U)
            self.lamg_warm = sol.value(self.opti.lam_g)

            u_sf = ptu.from_numpy(np.array(sol.value(self.U[:,0])).reshape((self.nu,1))[:,0])

            print(f"u delta: {ptu.from_numpy(u[:,0]) - u_sf}")

            return u_sf, ptu.from_numpy(self.X_warm)

def get_nominal_control_system_nodes(quad_params, ctrl_params, Ts):

    nominal_control_node_list = []

    def process_policy_input(x, r, c, radius=0.5):
        idx = [0,7,1,8,2,9]
        x_r, r_r = x[:, idx], r[:, idx]
        x_r = torch.clip(x_r, min=ptu.tensor(-3.), max=ptu.tensor(3.))
        r_r = torch.clip(r_r, min=ptu.tensor(-3.), max=ptu.tensor(3.))
        c_pos, c_vel = posVel2cyl.pytorch_vectorized(x_r, c, radius)
        return torch.hstack([r_r - x_r, c_pos, c_vel]), torch.hstack([x_r, c_pos, c_vel])
    process_policy_input_node = nm.system.Node(process_policy_input, ['X', 'R', 'Cyl'], ['ObsErr_CylPV', 'Obs_CylPV'], name='state_selector')
    nominal_control_node_list.append(process_policy_input_node)

    mlp = nm.modules.blocks.MLP(6 + 2, 3, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU,
                    hsizes=[20, 20, 20, 20]).to(ptu.device)
    policy_node = nm.system.Node(mlp, ['ObsErr_CylPV'], ['MLP_U'], name='mlp')
    nominal_control_node_list.append(policy_node)

    gravity_offset = lambda u: u + ptu.tensor([[0,0,-quad_params["hover_thr"]]])
    gravity_offset_node = nm.system.Node(gravity_offset, ['MLP_U'], ['MLP_U_grav'], name='gravity_offset')
    nominal_control_node_list.append(gravity_offset_node)

    pid = PID(Ts=Ts, bs=1, ctrl_params=ctrl_params, quad_params=quad_params)
    pid_node = nm.system.Node(pid, ['X', 'MLP_U_grav'], ['U', 'PID_X'], name='pid_control')
    nominal_control_node_list.append(pid_node)

    return nominal_control_node_list

class Predictor:

    def __init__(self, quad_params, ctrl_params, Tf_hzn, N, mlp_state_dict) -> None:

        self.r = ptu.tensor([[[2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]*N])
        self.c = ptu.tensor([[[1, 1]]*N])
        dts = generate_variable_timesteps(Ts, Tf_hzn, N)
        node_list = get_nominal_control_system_nodes(quad_params, ctrl_params, Ts)
        predictive_dynamics = lambda x, u, k: (euler(state_dot.pytorch_vectorized, x, u, dts[torch.round(k).long()], quad_params), k+1)
        predictive_dynamics_node = nm.system.Node(predictive_dynamics, input_keys=['X', 'U', 'K'], output_keys=['X', 'K'])
        node_list.append(predictive_dynamics_node)
        self.predictive_system = nm.system.System(node_list, nsteps=N)
        self.predictive_system.nodes[1].load_state_dict(mlp_state_dict)

    def check_violations(self, x_pred):
        # Extract x and y coordinates
        x = x_pred[0, :, 0]
        y = x_pred[0, :, 1]
        # Calculate the squared distances from the cylinder center
        distances = torch.sqrt((x - 1)**2 + (y - 1)**2) - 0.5
        # Check if any point is inside the cylinder (distance <= radius)
        outside_cylinder = torch.all(distances >= 0).int()
        print(distances)
        return outside_cylinder.unsqueeze(0).unsqueeze(0)
    
    def __call__(self, x, pid_state):
        self.predictive_system.nodes[3].callable.reset(torch.clone(pid_state))
        initial_conditions = {'X': x.unsqueeze(0), 'R':self.r, 'Cyl':self.c, 'K': ptu.tensor([[[0]]])}
        with torch.no_grad():
            predictions = self.predictive_system(initial_conditions, retain_grad=False, print_loop=False)
        violations = self.check_violations(predictions['X'])
        return predictions['X'][0], predictions['U'][0], violations

def run_wp_p2p(
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
    mlp_state_dict = torch.load(policy_save_path + 'wp_p2p_policy.pth')

    safety_filter = SafetyFilter(state_dot.casadi_vectorized, Ts, Tf_hzn_sf, N_sf, quad_params, euler)
    R = reference.waypoint('wp_p2p', average_vel=1.0)

    node_list = get_nominal_control_system_nodes(quad_params, ctrl_params, Ts)

    predictor = Predictor(quad_params, ctrl_params, Tf_hzn_pred, N_pred, mlp_state_dict)
    predictor_node = nm.system.Node(predictor, input_keys=['X', 'PID_X'], output_keys=['X_pred', 'U_pred', 'Violation'], name='predictor')
    node_list.append(predictor_node)

    filter = lambda x_seq, u_seq, violation: safety_filter(x_seq, u_seq, violation).unsqueeze(0)
    filter_node = nm.system.Node(filter, input_keys=['X_pred', 'U_pred', 'Violation'], output_keys=['U_filtered'], name='dynamics')
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

    safety_filter = SafetyFilter(state_dot.casadi_vectorized, Ts, Tf_hzn_sf, N_sf, quad_params, euler)
    R = reference.waypoint('wp_p2p', average_vel=1.0)

    node_list = get_nominal_control_system_nodes(quad_params, ctrl_params, Ts)

    # filter = lambda x_seq, u_seq, violation: safety_filter(x_seq, u_seq, violation).unsqueeze(0)
    def filter(x_seq, u_seq, violation):
        u, x_planned = safety_filter(x_seq, u_seq, violation)
        return u.unsqueeze(0), x_planned
    filter_node = nm.system.Node(filter, input_keys=['X', 'U', 'Violation'], output_keys=['U_filtered', 'X_planned'], name='dynamics')
    node_list.append(filter_node)

    # sys = mujoco_quad(state=quad_params["default_init_state_np"], quad_params=quad_params, Ti=Ti, Tf=Tf, Ts=Ts, integrator=integrator)
    sys = lambda x, u: x + state_dot.neuromancer(x, u, quad_params)*Ts
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
        'Violation': ptu.tensor([[[False]]*nsteps])
    }

    # set the mujoco simulation to the correct initial conditions
    # cl_system.nodes[5].callable.set_state(ptu.to_numpy(data['X'].squeeze().squeeze()))

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

    plot_mujoco_trajectories_wp_p2p([output], f'data/paper/dpcsf_adv_nav_{str(Ts)}.svg')

    state_prediction = ptu.to_numpy(output['X_planned']).transpose(1, 0, 2)

    animator = Animator(x_history, times, r_history, max_frames=500, save_path=media_save_path, state_prediction=state_prediction, drawCylinder=True)
    animator.animate()

    print('fin')

def run_nav_mj_no_event_trigger(
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

    safety_filter = SafetyFilter(state_dot.casadi_vectorized, Ts, Tf_hzn_sf, N_sf, quad_params, euler)
    R = reference.waypoint('wp_p2p', average_vel=1.0)

    node_list = get_nominal_control_system_nodes(quad_params, ctrl_params, Ts)

    # filter = lambda x_seq, u_seq, violation: safety_filter(x_seq, u_seq, violation).unsqueeze(0)
    def filter(x_seq, u_seq, violation):
        u, x_planned = safety_filter(x_seq, u_seq, violation)
        return u.unsqueeze(0), x_planned
    filter_node = nm.system.Node(filter, input_keys=['X', 'U', 'Violation'], output_keys=['U_filtered', 'X_planned'], name='dynamics')
    node_list.append(filter_node)

    sys = mujoco_quad(state=quad_params["default_init_state_np"], quad_params=quad_params, Ti=Ti, Tf=Tf, Ts=Ts, integrator=integrator)
    sys_node = nm.system.Node(sys, input_keys=['X', 'U_filtered'], output_keys=['X'], name='dynamics')
    node_list.append(sys_node)

    cl_system = nm.system.System(node_list, nsteps=nsteps)

    # load the pretrained policies
    cl_system.nodes[1].load_state_dict(mlp_state_dict)

    X = ptu.from_numpy(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32))
    X[:,:,7] = 1.6 # xdot adversarial
    X[:,:,8] = 1.6 # ydot adversarial

    # X[:,:,0] = 0.45
    # X[:,:,1] = 0.45
    data = {
        'X': X,
        'R': torch.concatenate([ptu.from_numpy(R(1)[None,:][None,:].astype(np.float32))]*nsteps, axis=1),
        'Cyl': torch.concatenate([ptu.tensor([[[1,1]]])]*nsteps, axis=1),
        'Violation': ptu.tensor([[[False]]*nsteps])
    }

    # set the mujoco simulation to the correct initial conditions
    cl_system.nodes[5].callable.set_state(ptu.to_numpy(data['X'].squeeze().squeeze()))

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

    plot_mujoco_trajectories_wp_p2p([output], f'data/paper/dpcsf_adv_nav_{str(Ts)}.svg')

    state_prediction = ptu.to_numpy(output['X_planned']).transpose(1, 0, 2)

    animator = Animator(x_history, times, r_history, max_frames=500, save_path=media_save_path, state_prediction=state_prediction, drawCylinder=True, perspective='topdown')
    animator.animate()

    print('fin')

def run_nav_mj(
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

    # safe set violation detection
    log = torch.load('large_data/nav_training_data.pt')
    bf = BarrierFunction(log)
    safety_filter = SafetyFilter(state_dot.casadi_vectorized, Ts, Tf_hzn_sf, N_sf, quad_params, euler, bf=bf)
    R = reference.waypoint('wp_p2p', average_vel=1.0)

    node_list = get_nominal_control_system_nodes(quad_params, ctrl_params, Ts)

    # in_safe_set = lambda x, bf=bf: ptu.tensor(bf(ptu.to_numpy(x.flatten()[0:6]))).unsqueeze(0).unsqueeze(0) == False
    def in_safe_set(obs, bf=bf):
        safe = ptu.tensor(bf(ptu.to_numpy(obs.flatten()[0:6]))).unsqueeze(0).unsqueeze(0)
        violation = safe == False
        if violation == True:
            print('fin')
        return violation
    in_safe_set_node = nm.system.Node(in_safe_set, input_keys=['Obs_CylPV'], output_keys=['Violation'])
    node_list.append(in_safe_set_node)

    def filter(x_seq, u_seq, violation):
        u, x_planned = safety_filter(x_seq, u_seq, violation)
        return u.unsqueeze(0), x_planned
    filter_node = nm.system.Node(filter, input_keys=['X', 'U', 'Violation'], output_keys=['U_filtered', 'X_planned'], name='dynamics')
    node_list.append(filter_node)

    sys = mujoco_quad(state=quad_params["default_init_state_np"], quad_params=quad_params, Ti=Ti, Tf=Tf, Ts=Ts, integrator=integrator)
    sys_node = nm.system.Node(sys, input_keys=['X', 'U_filtered'], output_keys=['X'], name='dynamics')
    node_list.append(sys_node)

    cl_system = nm.system.System(node_list, nsteps=nsteps)

    # load the pretrained policies
    cl_system.nodes[1].load_state_dict(mlp_state_dict)

    # X = ptu.from_numpy(quad_params["default_init_state_np"][None,:][None,:].astype(np.float32))
    # X[:,:,7] = 1.6 # xdot adversarial
    # X[:,:,8] = 1.6 # ydot adversarial

    X = ptu.tensor([[[  1.       ,   -1.       ,  -0.       ,   1.       ,
           0.       ,  -0.       ,   0.       ,   0,
          2.25,  -0.       ,   0.       ,  -0.       ,
           0.       , 522.98474  , 522.98474  , 522.98474  ,
         522.98474  ]]])

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

    plot_mujoco_trajectories_wp_p2p([output], f'data/paper/dpcsf_adv_nav_{str(Ts)}.svg')

    state_prediction = ptu.to_numpy(output['X_planned']).transpose(1, 0, 2)

    animator = Animator(x_history, times, r_history, max_frames=500, save_path=media_save_path, state_prediction=state_prediction, drawCylinder=True, perspective='topdown')
    animator.animate()

    print('fin')

def run_nav_mj_many(
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

    # safe set violation detection
    log = torch.load('large_data/nav_training_data.pt')
    bf = BarrierFunction(log)
    safety_filter = SafetyFilter(state_dot.casadi_vectorized, Ts, Tf_hzn_sf, N_sf, quad_params, euler, bf=bf)
    R = reference.waypoint('wp_p2p', average_vel=1.0)

    node_list = get_nominal_control_system_nodes(quad_params, ctrl_params, Ts)

    # in_safe_set = lambda x, bf=bf: ptu.tensor(bf(ptu.to_numpy(x.flatten()[0:6]))).unsqueeze(0).unsqueeze(0) == False
    @time_function
    def in_safe_set(obs, bf=bf):
        safe = ptu.tensor(bf(ptu.to_numpy(obs.flatten()[0:6]))).unsqueeze(0).unsqueeze(0)
        violation = safe == False
        if violation == True:
            print('fin')
        return violation
    in_safe_set_node = nm.system.Node(in_safe_set, input_keys=['Obs_CylPV'], output_keys=['Violation'])
    node_list.append(in_safe_set_node)

    def filter(x_seq, u_seq, violation):
        u, x_planned = safety_filter(x_seq, u_seq, violation)
        return u.unsqueeze(0), x_planned
    filter_node = nm.system.Node(filter, input_keys=['X', 'U', 'Violation'], output_keys=['U_filtered', 'X_planned'], name='dynamics')
    node_list.append(filter_node)

    sys = mujoco_quad(state=quad_params["default_init_state_np"], quad_params=quad_params, Ti=Ti, Tf=Tf, Ts=Ts, integrator=integrator)
    sys_node = nm.system.Node(sys, input_keys=['X', 'U_filtered'], output_keys=['X'], name='dynamics')
    node_list.append(sys_node)

    cl_system = nm.system.System(node_list, nsteps=nsteps)

    # load the pretrained policies
    cl_system.nodes[1].load_state_dict(mlp_state_dict)

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
    start_time = time.time()
    for data in tqdm(datasets):
        # set the mujoco simulation to the correct initial conditions
        cl_system.nodes[6].callable.set_state(ptu.to_numpy(data['X'].squeeze().squeeze()))
        cl_system.nodes[3].callable.reset(None)
        outputs.append(cl_system.forward(data, retain_grad=False))

    end_time = time.time()
    total_time = (end_time - start_time)
    average_time = total_time / npoints

    print("Average Time: {:.2f} seconds".format(average_time))
    x_histories = [ptu.to_numpy(outputs[i]['X'].squeeze()) for i in range(npoints)]
    u_histories = [ptu.to_numpy(outputs[i]['U'].squeeze()) for i in range(npoints)]
    r_histories = [np.vstack([R(1)]*(nsteps+1))]*npoints

    plot_mujoco_trajectories_wp_p2p(outputs, 'data/paper/dpc_adv_nav.svg')

    average_cost = np.mean([calculate_mpc_cost(x_history, u_history, r_history) for (x_history, u_history, r_history) in zip(x_histories, u_histories, r_histories)])

    print("Average MPC Cost: {:.2f}".format(average_cost))

    np.savez(
        file = f"data/dpc_sf_adv_nav_mj_{str(Ts)}.npz",
        x_history0 = ptu.to_numpy(outputs[0]['X'].squeeze()),
        u_history0 = ptu.to_numpy(outputs[0]['U'].squeeze()),
        x_history1 = ptu.to_numpy(outputs[1]['X'].squeeze()),
        u_history1 = ptu.to_numpy(outputs[1]['U'].squeeze()),
        x_history2 = ptu.to_numpy(outputs[2]['X'].squeeze()),
        u_history2 = ptu.to_numpy(outputs[2]['U'].squeeze()),
        x_history3 = ptu.to_numpy(outputs[3]['X'].squeeze()),
        u_history3 = ptu.to_numpy(outputs[3]['U'].squeeze()),
        x_history4 = ptu.to_numpy(outputs[4]['X'].squeeze()),
        u_history4 = ptu.to_numpy(outputs[4]['U'].squeeze()),
    )

    print('fin')


def run_nav_mj_many2(
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

    # safe set violation detection
    log = torch.load('large_data/nav_training_data.pt')
    bf = BarrierFunction(log)
    safety_filter = SafetyFilter(state_dot.casadi_vectorized, Ts, Tf_hzn_sf, N_sf, quad_params, euler, bf=bf)
    R = reference.waypoint('wp_p2p', average_vel=1.0)

    node_list = get_nominal_control_system_nodes(quad_params, ctrl_params, Ts)

    # in_safe_set = lambda x, bf=bf: ptu.tensor(bf(ptu.to_numpy(x.flatten()[0:6]))).unsqueeze(0).unsqueeze(0) == False
    @time_function
    def in_safe_set(obs, bf=bf):
        safe = ptu.tensor(bf(ptu.to_numpy(obs.flatten()[0:6]))).unsqueeze(0).unsqueeze(0)
        violation = safe == False
        if violation == True:
            print('fin')
        return violation
    in_safe_set_node = nm.system.Node(in_safe_set, input_keys=['Obs_CylPV'], output_keys=['Violation'])
    node_list.append(in_safe_set_node)

    def filter(x_seq, u_seq, violation):
        u, x_planned = safety_filter(x_seq, u_seq, violation)
        return u.unsqueeze(0), x_planned
    filter_node = nm.system.Node(filter, input_keys=['X', 'U', 'Violation'], output_keys=['U_filtered', 'X_planned'], name='dynamics')
    node_list.append(filter_node)

    sys = mujoco_quad(state=quad_params["default_init_state_np"], quad_params=quad_params, Ti=Ti, Tf=Tf, Ts=Ts, integrator=integrator)
    sys_node = nm.system.Node(sys, input_keys=['X', 'U_filtered'], output_keys=['X'], name='dynamics')
    node_list.append(sys_node)

    cl_system = nm.system.System(node_list, nsteps=nsteps)

    # load the pretrained policies
    cl_system.nodes[1].load_state_dict(mlp_state_dict)

    npoints = 5  # Number of points you want to generate in 2 dimensions ie. 5 == (5x5 grid)
    xy_values = ptu.from_numpy(np.linspace(-1, 1, npoints))  # Generates 'npoints' values between -10 and 10 for x
    z_values = ptu.from_numpy(np.linspace(-1, 1, 1))
    z_values = [ptu.tensor(0.)]

    datasets = []
    for xy in xy_values:
        v = 0 # 2.25 # directly towards the cylinder
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
    start_time = time.time()
    for data in tqdm(datasets):
        # set the mujoco simulation to the correct initial conditions
        cl_system.nodes[6].callable.set_state(ptu.to_numpy(data['X'].squeeze().squeeze()))
        cl_system.nodes[3].callable.reset(None)
        outputs.append(cl_system.forward(data, retain_grad=False))

    end_time = time.time()
    total_time = (end_time - start_time)
    average_time = total_time / npoints

    print("Average Time: {:.2f} seconds".format(average_time))
    x_histories = [ptu.to_numpy(outputs[i]['X'].squeeze()) for i in range(npoints)]
    u_histories = [ptu.to_numpy(outputs[i]['U'].squeeze()) for i in range(npoints)]
    r_histories = [np.vstack([R(1)]*(nsteps+1))]*npoints

    plot_mujoco_trajectories_wp_p2p(outputs, 'data/paper/dpc_adv_nav.svg')

    average_cost = np.mean([calculate_mpc_cost(x_history, u_history, r_history) for (x_history, u_history, r_history) in zip(x_histories, u_histories, r_histories)])

    print("Average MPC Cost: {:.2f}".format(average_cost))

    np.savez(
        file = f"data/dpc_sf_adv_nav_mj_{str(Ts)}.npz",
        x_history0 = ptu.to_numpy(outputs[0]['X'].squeeze()),
        u_history0 = ptu.to_numpy(outputs[0]['U'].squeeze()),
        x_history1 = ptu.to_numpy(outputs[1]['X'].squeeze()),
        u_history1 = ptu.to_numpy(outputs[1]['U'].squeeze()),
        x_history2 = ptu.to_numpy(outputs[2]['X'].squeeze()),
        u_history2 = ptu.to_numpy(outputs[2]['U'].squeeze()),
        x_history3 = ptu.to_numpy(outputs[3]['X'].squeeze()),
        u_history3 = ptu.to_numpy(outputs[3]['U'].squeeze()),
        x_history4 = ptu.to_numpy(outputs[4]['X'].squeeze()),
        u_history4 = ptu.to_numpy(outputs[4]['U'].squeeze()),
    )

    print('fin')

if __name__ == "__main__":

    # example usage
    import utils.pytorch as ptu
    from utils.integrate import euler, RK4

    ptu.init_dtype()
    ptu.init_gpu(use_gpu=False)

    Ti, Tf, Ts = 0.0, 10.0, 0.001
    N_sf = 30
    N_pred = None
    Tf_hzn_sf, Tf_hzn_pred = 0.50, None # 0.75 works, is best that does
    integrator = euler

    run_nav_mj_many(Ti, Tf, Ts, N_sf, N_pred, Tf_hzn_sf, Tf_hzn_pred, integrator)
    # run_nav_mj(Ti, Tf, Ts, N_sf, N_pred, Tf_hzn_sf, Tf_hzn_pred, integrator)
    # run_nav_mj_no_event_trigger(Ti, Tf, Ts, N_sf, N_pred, Tf_hzn_sf, Tf_hzn_pred, integrator)
    # run_wp_p2p(Ti, Tf, Ts, N_sf, N_pred, Tf_hzn_sf, Tf_hzn_pred, integrator)


    # quad_params = get_quad_params()

    # safety_filter = SafetyFilter(state_dot.casadi_vectorized, Ts, Tf_hzn_sf, N_sf, quad_params, integrator)
    # 
    # def example_usage():
    #     u_seq = np.array([[1., 1., 1., 1.]]*N)
    #     x_seq = np.vstack([quad_params["default_init_state_np"]]*(N+1))
    #     passthrough = safety_filter(u_seq, x_seq, True)
    #     computed = safety_filter(u_seq, x_seq, False)

    

    print('fin')
