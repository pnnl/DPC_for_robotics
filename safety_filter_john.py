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

    """
    The time-invariant variable-timestep robust predictive safety filter implementation
    """

    def __init__(self, dynamics, Ts, Tf_hzn, N, quad_params, integrator):

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

        self.nx = 17
        self.nu = 4
        delta = 0.00005 # Small robustness margin for constraint tightening
        alpha = 10000.0  # Gain term required to be large as per Wabersich2022 paper
        alpha2 = 1.0 # Multiplication factor for feasibility parameters
        Lh_x, Ld, Lf_x, Lhf_x = 0, 0, 0, 0

        # Define Box Constraints
        # -------------------------
        state_constraints = {}
        input_constraints = {}

        umax = [quad_params["maxCmd"]*100]  #model.umax*np.ones((nu,1)) # Input constraint parameters(max)
        umin = [quad_params["minCmd"]*100] #model.umin*np.ones((nu,1)) # Input constraint parameters (min)

        scu_k = quad_params["state_ub"] 
        scl_k = quad_params["state_lb"]
        scu_k[0:13] *= 100
        scl_k[0:13] *= 100
        
        state_constraints[0] = lambda x, k: x - scu_k
        state_constraints[1] = lambda x, k: scl_k - x
        # cylinder constraint
        x_c, y_c, r = 1.0, 1.0, 0.51
        state_constraints[2] = lambda x, k: r**2 * (1 + k * dts[k] * 0.001) - (x[0] - x_c)**2 - (x[1] - y_c)**2
        # state_constraints[2] = lambda x, k: r**2 - (x[0] - x_c)**2 - (x[1] - y_c)**2 # without expanding cylinder term

        # Quadratic CBF
        cbf_const = (np.sqrt(2) - 0.5) ** 2
        # terminal constraint set
        xr = np.copy(quad_params["default_init_state_np"])
        xr[0], xr[1], xr[2] = 2, 2, 1 # x, y, z, (z is flipped)
        hf = lambda x: (x-xr).T @ (x-xr) - cbf_const

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
        self.s_opts = {"max_iter": 300, "print_level": 1, "tol": 1e-5}

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

        # Terminal constraint, enforced at last time step
        rob_marg_term = Lhf_x * Ld * Lf_x ** (self.N - 1)
        self.opti_feas.subject_to(hf(self.X_f[:, -1]) <= self.XiN_f - rob_marg_term)
        self.opti.subject_to(hf(self.X[:, -1]) <= self.XiN - rob_marg_term)

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

    def __call__(self, u_seq, x_seq, constraints_satisfied):

        u_seq = u_seq.T
        x_seq = x_seq.T

        if constraints_satisfied:

            # Save solution for warm start
            self.X_warm = x_seq
            self.U_warm = u_seq
            self.Xf_warm = x_seq
            self.Uf_warm = u_seq
            self.Xi_warm = np.zeros((self.nc, N))
            self.XiN_warm = 0.0
            return u_seq[:,0]
        
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

            # try:
            #     self.lamgf_warm = feas_sol.value(self.opti_feas.lam_g) # dual variables
            # except:
            #     print('initialized no dual variables yet for feasibility problem')

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
            self.opti.set_initial(self.opti.lam_g, self.lamgf_warm[:-((self.N)*self.nc+1)]) # Removing lambdas associated with non-negative constraint in feasibility problem

            try:
                sol = self.opti.solve()
            except:
                print(traceback.format_exc())
                print('------------------------------------INFEASIBILITY---------------------------------------------')
                return self.opti.debug, feas_sol.value(self.Xi_f), feas_sol.value(self.XiN_f)
            
            # Save solution for warm start
            self.X_warm = sol.value(self.X)
            self.U_warm = sol.value(self.U)

            try:
                self.lamg_warm = sol.value(self.opti.lam_g)
            except:
                print('initialized, no dual variables')

            return np.array(sol.value(self.U[:,0])).reshape((self.nu,1))[:,0]

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

    

if __name__ == "__main__":

    # example usage
    import utils.pytorch as ptu
    from utils.integrate import euler, RK4

    ptu.init_dtype()
    ptu.init_gpu()

    Ti, Tf, Ts = 0.0, 3.5, 0.001
    N = 100
    Tf_hzn = 1.0
    integrator = 'euler'
    quad_params = get_quad_params()

    safety_filter = SafetyFilter(state_dot.casadi_vectorized, Ts, Tf_hzn, N, quad_params, euler.numpy)
    
    def example_usage():
        u_seq = np.array([[1., 1., 1., 1.]]*N)
        x_seq = np.vstack([quad_params["default_init_state_np"]]*(N+1))
        passthrough = safety_filter(u_seq, x_seq, True)
        computed = safety_filter(u_seq, x_seq, False)

    

    print('fin')