from casadi import *
import matplotlib.pyplot as plt
from neuromancer import psl
import torch
from closed_loop import DPC_control
from closed_loop import DummyControl
from closed_loop import ClosedLoopSystem
from closed_loop import NNModel
import copy
from adaptive_estimator.adaptive_estimator import AdaptiveEstimator
import random

class NNAdaptive():
    def __init__(self, NNx, NNu, nx, nu, W=0):
        self.NNx = NNx
        self.NNu = NNu
        self.umax = np.array([1, 1])
        self.umin = np.array([0, 0])
        self.xmax = [1., 1.]
        self.xmin = [0., 0.]
        self.x0 = [0.1, 0.1]
        self.dt = 1.0
        self.nregressor = nx + 1  # TO UPDATE: nregressor should be the same as ntheta, consolidate
        self.nf = nx
        self.nx = nx
        self.nu = nu
        self.ntheta = nx + 1
        self.W = W
        self.theta = np.zeros((nx**2+nx,1))


    def __call__(self, x, k, u):
        '''Forward dynamics of the system based on the model parameters
        x: current state
        k: time
        u: control
        theta: model parameter
        '''

        xnext = copy.deepcopy(x)

        theta_mat = self.theta_mat(self.theta)
        fnn = self.NNx(x, 0, u).reshape((nx, 1))
        phi = np.vstack((fnn, 1.0))
        xnext = np.matmul(theta_mat.T, phi)

        return xnext
    def f(self, x, xkp1):
        ''' This is the output function for the adaptive estimator. Note this needs to take into account the discretization used, if any.
               x: current state
               xkp1: next state
               '''

        return xkp1

    def regressor(self, x, k, u):
        '''This is the nonlinear component of the dynamics in the form of f = theta^T phi(x), where phi is the regressor, f is the output and theta is the mode parameter matrix
                x: current state
                '''

        fnn = self.NNx(x, k, u).reshape((nx, 1))
        phi = np.vstack((fnn, 1.0))

        m = sqrt( 1.0 + phi.T.dot(phi) )
        psi = phi/m

        return psi, m


    def model_error(self, x, theta_error):
        ''' Computes worst case model error based on current conservative error of model parameters'''


        return np.nan

    def theta_mat(self, theta):
        '''This converts the list/vector form of the model parameters to the matrix form of the model parameters
                theta: list/vector from of model parameters
                '''

        theta_mat = theta.reshape((self.nregressor, self.nx))

        return theta_mat

    def theta_demat(self, theta_mat):
        '''This converts the matrix form of the model parameters to the list/vector form
                theta_mat: Matrix form of model parameters'''
        return theta_mat.reshape((-1, 1))




class TwoTankDynamics():
    def __init__(self, model,W=0):
        self.model = model
        self.c1 = model.params[2]['c1']  # inlet valve coefficient
        self.c2 = model.params[2]['c2']   # tank outlet coefficient
        self.c1_max = 0.12
        self.c1_min = 0.07
        self.c2_max = 0.08
        self.c2_min = 0.03
        self.umax = np.array([1, 1])
        self.umin = np.array([0, 0])
        self.xmax = [1., 1.]
        self.xmin = [0., 0.]
        self.x0 = [0.1, 0.1]
        self.W = W

    def __call__(self, x, k, u):

        dhdt = copy.deepcopy(x)
        dhdt[0] = self.c1 * (1.0 - u[1]) * u[0] - self.c2* sqrt(x[0]) + self.W#*sin(0.2*k)
        dhdt[1] = self.c1 * u[1] * u[0] + self.c2 * sqrt(x[0]) - self.c2 * sqrt(x[1])

        return dhdt


if __name__ == "__main__":



    """
    # # #  Arguments, dimensions, bounds
    """
    Nsim = 1000

    ## perturbation
    W = 0.0

    # ground truth system model
    gt_model = psl.nonautonomous.TwoTank()
    model = TwoTankDynamics(gt_model, W=W)
    nx = gt_model.nx
    ny = gt_model.nx
    nu = gt_model.nu
    nmodel = 2



    # Model bounds
    c1_max = model.c1_max
    c1_min = model.c1_min
    c2_max = model.c2_max
    c2_min = model.c2_min
    model_max = torch.tensor([[c1_max, c2_max]])
    model_min = torch.tensor([[c1_min, c2_min]])

    seed = 4

    #---------------------------
    # Load system ID model
    model_name = 'two_tank_model.pth'
    params = {'x_input_key':'xn', 'u_input_key':'U'}
    sysid_model = NNModel(nx=nx, nu=nu, modelpth=model_name, params=params)

    #----------------------------
    # Reference trajectory
    r_nsteps = Nsim
    np_refs = psl.signals.step(r_nsteps + 1, 1, min=0.2, max=model.xmax[0], randsteps=1, rng=np.random.default_rng(seed=seed))
    R = torch.tensor(np_refs, dtype=torch.float32).reshape(  1, r_nsteps + 1)
    torch_ref = torch.cat([R, R], dim=0)

    #-------------------------------
    # Load DPC policies

    # Nominal
    policy_name = 'two_tank'
    version = 4
    policy = torch.load(policy_name+'_policy_'+str(version)+'.pth')
    policy_params = torch.load(policy_name + '_params_'+str(version)+'.pth')
    u_DPC = DPC_control(nu, policy, policy_params, {'r': torch_ref}, umin=model.umin, umax=model.umax)
    dt = policy_params['ts']

    #-------------------------------
    # Construct adaptive model based on NNs
    model_adaptive = NNAdaptive(NNx=sysid_model, NNu=u_DPC, nx=nx, nu=nu, W = 0)

    #----------------------------------
    # Setup Adaptive Estimator
    theta_bounds = []
    theta0 = np.vstack((np.eye(nx).reshape((-1,1)), np.zeros((nx, 1))))
    theta_max_error = 0.5
    theta_bounds.append( lambda theta, theta0=theta0: np.matmul((theta - theta0).T, (theta - theta0)) - theta_max_error )
    cl_params = {'Gamma': 0.05, 'xi_NG': 0.001, 'xi_CL': 0.3, 'w': W}

    # ---------------------------------
    # Setup and run simulation


    dummy_control = DummyControl()

    Ntotal_sim = 100
    total_error_NN = []
    total_error_adaptive = []
    random.seed(1)
    for kk in range(Ntotal_sim):

        print('kk = '+str(kk))

        # Ramdomize nominal control and initial condition
        x01 = random.uniform(0.0, 1.0)
        x02 = random.uniform(0.0, 1.0)
        x0 = np.array([x01, x02]).reshape((nx, 1))  # 4.0 + model.x0.reshape((nx, 1)) # 23.5*np.ones((nx,1)) #

        a1 = random.uniform(-0.1, 0.1) #0.1#
        b1 = random.uniform(np.abs(a1), 1.0 - np.abs(a1)) #0.5#
        w1 = random.uniform(-0.1, 0.1)
        a2 = random.uniform(-0.1, 0.1) #0.1#
        b2 = random.uniform(np.abs(a2), 1.0 - np.abs(a2)) #0.5#
        w2 =  random.uniform(-0.1, 0.1)
        print('a1 = '+str(a1))
        print('b1 = ' + str(b1))
        print('w1 = ' + str(w1))
        print('a2 = ' + str(a2))
        print('b2 = ' + str(b2))
        print('w2 = ' + str(w2))
        u_nom = lambda x, k: np.array([a1 * sin(w1*k) + b1, a2 * sin(w2*k) + b2])
        #u_nom = lambda x, k: np.array([a1 * sin(x[0]) + b1, a2 * sin(x[1]) + b2])

        CL_nom = ClosedLoopSystem(f=model, u=dummy_control, u_nom=u_nom, dt=dt, int_type=policy_params['integrator_type'])
        CL_NN = ClosedLoopSystem(f=sysid_model, u=dummy_control, u_nom=u_nom, dt=dt, int_type='cont')
        model_adaptive.NNu = u_nom

        # Run simulation with true system and NN-based sysid model
        [X_nom, U_nom] = CL_nom.simulate(x0=x0, N=Nsim)
        [X_NN, U_NN] = CL_NN.simulate(x0=x0, N=Nsim)

        if np.isnan(np.sum(X_nom)):
            print('Found nan, moving to next iteration...')
            continue

        # Run adaptive estimator
        AE = AdaptiveEstimator(theta_val=theta0, f=model_adaptive, window=100,
                               Theta=theta_bounds, nx=nx, nu=nu, dT=dt, params=cl_params)

        theta_est = []
        ranks = []
        for ii, xi in enumerate(X_nom):
            ui = U_nom[ii,:]

            ae_vals = AE(xi, ii, ui)
            theta_est.append(ae_vals[0])
            ranki = AE.check_regressor_rank()
            ranks.append(ranki)

        # Run simulation with adapted model parameters
        theta_est = np.array(theta_est).reshape((-1, nx**2+nx))
        theta_new = AE.theta_val
        model_adaptive.theta = theta_new

        CL_adaptive = ClosedLoopSystem(f=model_adaptive, u=dummy_control, u_nom=u_nom, dt=dt, int_type='cont')
        [X_adapt, U_adapt] = CL_adaptive.simulate(x0=x0, N=Nsim)

        # Compute errors per kk iteration
        NNerror = X_nom - X_NN
        total_error_NN.append( np.linalg.norm(NNerror) )
        Adaptive_error = X_nom - X_adapt
        total_error_adaptive.append( np.linalg.norm(Adaptive_error) )

        if total_error_NN[-1] < total_error_adaptive[-1]:
            plot_me = True
        else:
            plot_me = False


        # Plots

        if kk < 4 or plot_me:
            t = [kk for kk in range(Nsim + 1)]
            plt.figure()
            plt.plot(np_refs, color='navy', label='ref', linewidth=3.0)
            plt.plot(X_nom[:, 0], color='crimson', label='x1 (nominal)', linewidth=2.0)
            plt.plot(X_nom[:, 1], color='chartreuse', label='x2 (nominal)', linewidth=2.0)
            plt.plot(X_NN[:, 0], color='orange', linestyle='--', label='x1 (sysid)', linewidth=2.0)
            plt.plot(X_NN[:, 1], color='magenta', linestyle='--', label='x2 (sysid)', linewidth=2.0)
            plt.xlabel('Time step')
            plt.ylabel('Tank Level')
            plt.title('k = '+str(kk))
            plt.legend(loc='lower left')

            plt.figure()
            plt.plot(np_refs, color='navy', label='ref', linewidth=3.0)
            plt.plot(X_nom[:, 0], color='crimson', label='x1 (nominal)', linewidth=2.0)
            plt.plot(X_nom[:, 1], color='chartreuse', label='x2 (nominal)', linewidth=2.0)
            plt.plot(X_adapt[:, 0], color='orange', linestyle='--', label='x1 (adaptive)', linewidth=2.0)
            plt.plot(X_adapt[:, 1], color='magenta', linestyle='--', label='x2 (adaptive)', linewidth=2.0)
            plt.xlabel('Time step')
            plt.ylabel('Tank Level')
            plt.title('k = ' + str(kk))
            plt.legend(loc='lower left')

        if 0:
            plt.figure()
            for ii in range(nu):
                plt.step(range(U_nom.shape[0]), 1.0 * np.ones(U_nom.shape[0]), 'k', linewidth=3.0)
                plt.step(range(U_nom.shape[0]), 0.0 * np.ones(U_nom.shape[0]), 'k', linewidth=3.0)
            plt.plot(U_nom[:, 0], color='crimson', label="u1 (DPC nominal)", linewidth=2.0)
            plt.plot(U_nom[:, 1], color='chartreuse', label="u2 (DPC nominal)", linewidth=2.0)
            plt.plot(U_NN[:, 0], color='orange', linestyle='--', label="u1 (DPC adaptive)", linewidth=2.0)
            plt.plot(U_NN[:, 1], color='magenta', linestyle='--', label="u2 (DPC adaptive)", linewidth=2.0)
            plt.xlabel('Time step')
            plt.ylabel('Pump and Valve (Inputs)')
            plt.legend(loc='upper right', bbox_to_anchor=(0.77, 0.95))

            plt.figure()
            plt.plot(theta_est, label='theta_est')
            plt.xlabel('Time step')
            plt.ylabel('Values')
            plt.legend()

    plt.figure()
    plt.plot(total_error_NN, label='sysID')
    plt.plot(total_error_adaptive, label='adaptive')
    plt.xlabel('Iteration k')
    plt.ylabel('Cumulative trajectory error')
    plt.legend()
    plt.show()