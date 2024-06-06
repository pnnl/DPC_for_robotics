import copy

from casadi import *
import numpy as np
import traceback
from neuromancer.modules import blocks
import torch
from tqdm import tqdm
from neuromancer.modules.activations import activations
import matplotlib.pyplot as plt
from safetyfilter import integrate_adaptive

class AdaptiveEstimator():
    def __init__(self, theta_val, f, type='LSE', window=20, Theta=None, nx=1, nu=1, integrator_type='cont', dT = 0.1, cl_params={}):

        self.theta_val = np.array(theta_val)
        self.f = f
        self.type = type
        self.window = window
        self.Theta = Theta
        self.nx = nx
        self.nu = nu
        self.ntheta = len(theta_val)
        self.integrator_type = integrator_type
        self.dT = dT
        self.cl_params = cl_params

        self.Ldk = np.nan


        # Setup window of past trajectories
        self.x = []
        self.u = []

        # Setup windows of trajectories that are persistently exciting
        self.Z = []
        self.M = []
        self.F = []

        # Define estimator to use
        self.estimator = {'LSE': self.LSE, 'ConcurrentLearning': self.concurrent_learning}

        # Initialize rank test to check if enough persistently exciting data points have been added.
        self.rank_test = False

        # Set CL parameters
        if self.cl_params:
            self.Gamma = self.cl_params['Gamma']
            self.xi_NG = self.cl_params['xi_NG']
            self.xi_CL = self.cl_params['xi_CL']
            self.w = self.cl_params['w']
        else:
            self.Gamma = 0.05
            self.xi_NG = 0.001
            self.xi_CL = 0.3
            self.w = 0.0

        # Define casadi optimization parameters
        self.p_opts = {"expand": True, "print_time": False, "verbose": False}
        self.s_opts = {'max_iter': 300,
                       'print_level': 1,
                       'warm_start_init_point': 'yes',
                       'tol': 1e-8,
                       'constr_viol_tol': 1e-8,
                       "compl_inf_tol": 1e-8,
                       "acceptable_tol": 1e-8,
                       "acceptable_constr_viol_tol": 1e-8,
                       "acceptable_dual_inf_tol": 1e-8,
                       "acceptable_compl_inf_tol": 1e-8,
                       }

        if self.type == 'LSE':
            self.setup_LSE()
        if self.type == 'ConcurrentLearning':
            self.setup_CL()



    def setup_LSE(self):
        '''Setup parameters and optimization problem for LSE problem'''

        self.opti = Opti()
        self.X = self.opti.parameter(self.nx, self.window)
        self.Xhat = self.opti.parameter(self.nx, self.window)
        self.U = self.opti.parameter(self.nu, self.window)
        self.theta = self.opti.variable(self.ntheta, 1)

        # Define constraints on model parameters
        for Thetai in self.Theta:
            self.opti.subject_to(Thetai(self.theta) <= 0.0)

        # Set the solver
        self.opti.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend

    def setup_CL(self):
        '''Setup parameters for CL implementation'''

        # Compute worst case Lyapunov function for error between estimate and true parameters
        self.V0, self.theta_error0 = self.computeCLLyapunov()

        # Set current V value to initial, worst case estimate. Same with model error worst case
        self.V = self.V0
        self.theta_error = self.theta_error0



    def computeCLLyapunov(self):
        '''Solve for worst case Lyapunov function w.r.t error between true and approximate model parameters based on
        assumption that the model parameters must stay within constraints defined by Theta'''

        # Setup optimization
        opti = Opti()
        theta1 = opti.variable(self.ntheta, 1)
        theta2 = opti.variable(self.ntheta, 1)
        opti.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend

        # Define constraints on model parameters
        for Thetai in self.Theta:
            opti.subject_to(Thetai(theta1) <= 0.0)
            opti.subject_to(Thetai(theta2) <= 0.0)

        # Define cost
        Gamma_inv = 1.0/self.Gamma
        error_cost = Gamma_inv*(theta1 - theta2).T@(theta1 - theta2)
        opti.minimize(-error_cost)

        # Solve optimization
        sol = opti.solve()

        # Extract estimate of worst case Lyapunov function
        V0 = sol.value(error_cost)
        theta1_sol = sol.value(theta1)
        theta2_sol = sol.value(theta2)
        theta0 = np.linalg.norm(theta1_sol - theta2_sol)

        return V0, theta0



    def __call__(self, x, k, u):
        ''' Evaluate the estimator and handle new state and input values
        x: Current state
        k: Current time
        u: Current input
        '''

        # If the history of previous trajectories is not long enough, augment it
        if len(self.x) < self.window:

            self.x.append(x.tolist())
            self.u.append(u.tolist())

            return self.theta_val, self.Ldk

        else:

            # Replace oldest trajectory values with new updates
            self.x.append(x.tolist())
            self.x = self.x[1:]
            self.u.append(u.tolist())
            self.u = self.u[1:]

            # Evaluate the estimator
            return self.estimator[self.type](x,k,u)


    def LSE(self, x, k, u):
        '''Least squares estimator
        x: current state
        k: current time
        u: current input
        '''

        # Check rank condition
        ranki = self.check_regressor_rank(self.x, self.u)


        if self.rank_test == False:

            # Set parameters in opti as the previous trajectories
            self.opti.set_value( self.X, np.array(self.x).T)
            self.opti.set_value( self.U, np.array(self.u).T)

            # Construct LSE cost function
            cost = 0
            xk = self.X[:,0]
            l = k
            for ii in range(self.window):
                xi = self.X[:, ii]
                ui = self.U[:, ii]

                cost += (xi - xk).T@(xi - xk)

                l += 1
                xk = integrate_adaptive( self.f, xk, l, ui, self.theta, self.dT, integrator_type=self.integrator_type )

            self.opti.minimize( cost )

            # Warm start with previous value of  model parameters
            self.opti.set_initial(self.theta, self.theta_val)

            # Solve LSE problem
            sol = self.opti.solve()

            # Extract model parameter and estimate of worst disturbance bound
            theta_val = np.array([sol.value(self.theta)]).flatten()
            Ldk = sol.value(cost)

            # Check if the previous disturbance error was worse than the previous and update only if it improves
            #if (np.isnan(self.Ldk)) or (Ldk < self.Ldk):
            self.Ldk = Ldk
            self.theta_val = theta_val


        return self.theta_val, self.Ldk

    def concurrent_learning(self, x, k, u):
        '''Implements the concurrent learning algorithm from Djaneye-Boundjou et al 2020 paper to estimate the model parameters
        x: current state
        k: current time
        u: current input
        '''

        # Check that the expected control input matches that associated with the regressor matrix
        u_r = self.f.u(np.array(self.x[-2]).reshape((self.nx,1)),0)
        if np.linalg.norm(np.array(u).reshape((self.nu,1)) - u_r) > 1e-6:
            print('Resetting CL!')
            self.cl_reset()
            return self.theta_val, np.nan


        # Collect the stored window data
        self.x_cl = self.x

        # If the rank test has not been satisfied, update the stored data based on the window from self.x
        if self.rank_test == False:

            # Compute and store the concurrent learning data
            self.update_cl_data()

            # Check the rank condition
            ranki = self.check_regressor_rank()
            print('ranki = '+str(ranki))

            # If the rank condition is satisfied, set rank_test to true to avoid adding excessive data
            if ranki == self.f.nregressor:
                self.rank_test = True

        # If the rank test is satisfied, we do not need to keep adding to the stored data unless it is persistently exciting
        else:
            ranki = self.f.nregressor

            # Only update if the full rank condition still holds
            self.replace_cl_data()

        # Compute components of CL update law (gradient-based)
        psi, m = self.f.regressor(np.array(self.x[-2]))
        F = np.matmul(self.f.theta_mat(self.theta_val).T, psi*m)
        fi = self.f.f(np.array(self.x[-2]).reshape((self.nx,1)), np.array(self.x[-1]).reshape((self.nx,1)))
        q = F - fi
        ANG = np.matmul(psi, q.T)/m

        # Compute components of CL update law (data-based)
        Acl = np.zeros((self.f.ntheta, self.f.nf))
        for ii, fi in enumerate(self.F):

            psi_j = np.array(self.Z[ii]).reshape((self.f.nregressor,1))
            m = self.M[ii]
            F = np.matmul(self.f.theta_mat(self.theta_val).T, psi_j*m)
            qi = F - np.array(fi).reshape((self.f.nf,1))

            Acl += np.matmul(psi_j, qi.T)/m


        # Update the estimated model parameters (matrix form)
        theta_mat_new = self.f.theta_mat(self.theta_val) - self.Gamma*( self.xi_NG * ANG + self.xi_CL * Acl)

        # Convert matrix model parameters to list form
        self.theta_val = self.f.theta_demat(theta_mat_new)

        # Update Lyapunov estimate
        self.V, self.theta_error = self.updateLyapunov()

        # Compute worst case model error
        Ldk = self.f.model_error(x, self.theta_error)

        return self.theta_val, Ldk

    def updateLyapunov(self):
        '''Use current data set to compute the next Lyapunov value and error approximation (worst case) (see Djaneye-Boundjou2020)'''

        # Compute update terms
        eta = self.Gamma
        eps_CL = self.xi_CL
        eps_NG = self.xi_NG

        Z = np.array(self.Z).reshape((self.f.ntheta, -1))
        ZZt = np.matmul(Z, Z.T)

        psi, m = self.f.regressor(np.array(self.x[-2]))
        phi = psi * m
        Phi = np.matmul(phi,phi.T)

        eigv, eigvec = np.linalg.eig(Phi)
        l2 = max(eigv.real)

        eigv4, eigvec4 = np.linalg.eig(ZZt)
        l3 = min(eigv4.real)
        l4 = max(eigv4.real)

        R1 = 1.0 - eta*eps_CL*l3*( 2.0 - eta*( (2.0*eps_NG*l2)/(m**2) + eps_CL*l4 ) )

        P = np.eye(self.f.ntheta) - self.Gamma * self.xi_NG * Phi / (m ** 2) - self.Gamma * self.xi_CL * ZZt

        # Compute bounds for theta and V terms
        sum_psi = np.zeros((self.f.nregressor,1))
        for ii, Zi in enumerate(self.Z):
            psi_j = np.array(Zi).reshape((self.f.nregressor,1))
            m_j = self.M[ii]
            sum_psi += psi_j/m_j
        bound_psi = np.linalg.norm(sum_psi)
        bound_psi_k = np.linalg.norm(psi/m)
        eps_w_bar_bound = ( self.xi_NG*bound_psi_k + self.xi_CL*bound_psi )*self.w

        Pbound = np.linalg.norm(P)
        theta_error_new = min( Pbound*self.theta_error + self.Gamma*eps_w_bar_bound, self.theta_error)

        Ebound = Pbound*eps_w_bar_bound
        Ewbar_bound = self.Gamma*eps_w_bar_bound**2

        Vnew = R1 * self.V + 2.0*self.theta_error*Ebound + Ewbar_bound

        return Vnew, theta_error_new


    def check_regressor_rank(self):
        ''' Computes the rank of the stored regressor data '''

        Znp = np.array(self.Z)

        return np.linalg.matrix_rank(Znp)

    def compute_parameter_model(self, x):
        '''Outputs the estimated model output based on the current state
        x: current state
        '''
        theta_mat = self.f.theta_mat(self.theta_val)
        psi,m = self.f.regressor(x)

        F = np.matmul(theta_mat.T, psi*m )

        return F

    def check_convergence_condition(self):
        '''Checks if convergence condition as per Djaneye-Boundjou et al is met. If positive, convergence condition is satisfied '''

        if len(self.Z) > 0:
            Z = np.array(self.Z).reshape((self.f.ntheta, -1))

            psi = np.array(self.Z[-1]).reshape((self.f.nregressor,1))
            m = self.M[-1]
            phi = psi*m

            eigv, eigvec = np.linalg.eig( np.outer(phi, phi) )
            l2 = max(eigv)

            eigv4, eigvec4 = np.linalg.eig( np.matmul(Z, Z.T) )
            l4 = max(eigv4)

            return 2.0*m**2/( 2.0*self.xi_NG*l2 + self.xi_CL*l4*m**2 ) - self.Gamma

        else:
            return np.nan

    def update_cl_data(self):
        '''Updates data stored in F, M and Z used in the concurrent learning approach'''

        for ii, xi in enumerate(self.x_cl):

            if ii < len(self.x_cl) - 1:

                # Compute Z and M evaluations and store
                Zi, mi = self.f.regressor(np.array(xi).reshape((self.nx,1)))
                self.Z.append(Zi.flatten().tolist())
                self.M.append(mi)


                # Compute F evaluations and store
                xip1 = self.x_cl[ii+1]
                fi = self.f.f(np.array(xi), np.array(xip1))
                self.F.append( fi.flatten().tolist() )

                # Compute convergence ratio, goal is to maximize the conv_ratio
                Z = np.array(self.Z).reshape((self.f.ntheta, -1))
                ZZt = np.matmul(Z, Z.T)
                eigv, eigvec = np.linalg.eig(ZZt)
                l3 = min(eigv.real)
                l4 = max(eigv.real)
                self.conv_ratio = l3/l4


    def replace_cl_data(self):
        ''' If the recent state retains the rank of Z and improves convergence, update Z, M, F '''

        x = self.x_cl[-2]
        xp1 = self.x_cl[-1]

        # Compute Z and M evaluations
        Zi, mi = self.f.regressor(np.array(x).reshape((self.nx, 1)))

        # Compute new Z
        Znew = copy.deepcopy(self.Z)
        Znew.append(Zi.flatten().tolist())

        # Check rank and convergence condition
        Znew_np = np.array(Znew[-self.window:])
        rankZnew = np.linalg.matrix_rank(Znew_np)
        ZZt = np.matmul(Znew_np, Znew_np.T)
        eigv, eigvec = np.linalg.eig(ZZt)
        l3 = min(eigv.real)
        l4 = max(eigv.real)
        new_conv_ratio = l3/l4


        # If the rank condition is met and the convergence ratio is imporved, then update Z, M and F with new point
        if rankZnew == self.f.nregressor and new_conv_ratio > self.conv_ratio:

            # Update Z, M and F lists with new point
            self.Z = Znew[-self.window:]

            Mnew = copy.deepcopy(self.M)
            Mnew.append(mi)
            self.M = Mnew[-self.window:]

            fi = self.f.f(np.array(x), np.array(xp1))
            Fnew = copy.deepcopy(self.F)
            Fnew.append(fi)
            self.F = Fnew[-self.window:]

            # Update convergence ratio
            self.conv_ratio = new_conv_ratio

    def cl_reset(self):
        '''Reset the adaptive estimator using cl by removing all data fraom Z, M, F'''

        self.Z = []
        self.M = []
        self.F = []
        self.x = []
        self.u = []
        self.x_cl = []
        self.rank_test = False




