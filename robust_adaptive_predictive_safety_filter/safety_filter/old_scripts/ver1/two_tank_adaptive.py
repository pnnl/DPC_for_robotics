from casadi import *
import safetyfilter as sf
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from neuromancer import psl
import torch
from safetyfilter import DPC_control
from safetyfilter import DummyControl
from safetyfilter import ClosedLoopSystem
from safetyfilter import integrate
import copy
import time
from mpl_toolkits.mplot3d import Axes3D
import math
from adaptive_estimator import AdaptiveEstimator

class TwoTankDynamicsAdaptive():
    def __init__(self, model,dbar=0):
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
        # -------New try ---------
        self.nregressor = 2 #4
        # -------New try ---------
        self.nx = 2

    def __call__(self, x, k, u, theta):

        u = -x

        dhdt = copy.deepcopy(x)
        dhdt[0] = theta[0] * (1.0 - u[1]) * u[0] - theta[1]* sqrt(x[0])
        dhdt[1] = theta[0] * u[1] * u[0] + theta[1] * sqrt(x[0]) - theta[1] * sqrt(x[1])

        return dhdt

    def regressor(self, x, k, u):

        # phi = np.zeros(self.nregressor)
        # phi[0] = (1.0 - u[1])*u[0]
        # phi[1] = -sqrt(x[0])
        # phi[2] = u[0]*u[1]
        # phi[3] = sqrt(x[0]) - sqrt(x[1])

        #-------New try ---------
        phi = np.zeros(self.nregressor)
        u = -x
        phi[0] = (1.0 - u[1]) * u[0] + u[0] * u[1]
        phi[1] = -sqrt(x[0]) + sqrt(x[0]) - sqrt(x[1])
        # -------New try ---------

        m = sqrt( 100.0 + phi.dot(phi) )
        psi = phi/m

        return psi, m

    def theta_mat(self, theta):

        #theta_mat = np.zeros((self.nregressor, self.nx))

        # theta_mat[0, 0] = theta[0]
        # theta_mat[1, 0] = theta[1]
        # theta_mat[2, 1] = theta[0]
        # theta_mat[3, 1] = theta[1]

        # -------New try ---------
        theta_mat = np.zeros((self.nregressor, 1))
        theta_mat[0] = theta[0]
        theta_mat[1] = theta[1]
        # -------New try ---------

        return theta_mat

    def theta_demat(self, theta_mat):

        return np.array([theta_mat[0,0], theta_mat[1,0]])

    #def PE_condition(self, x, k, u):



class TwoTankDynamics():
    def __init__(self, model,dbar=0):
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


    def __call__(self, x, k, u):

        dhdt = copy.deepcopy(x)
        u = -x
        dhdt[0] = c1 * (1.0 - u[1]) * u[0] - c2* sqrt(x[0])
        dhdt[1] = c1 * u[1] * u[0] + c2 * sqrt(x[0]) - c2 * sqrt(x[1])

        return dhdt


if __name__ == "__main__":



    """
    # # #  Arguments, dimensions, bounds
    """
    Nsim = 2000

    # ground truth system model
    gt_model = psl.nonautonomous.TwoTank()
    model = TwoTankDynamics(gt_model)
    model_adaptive = TwoTankDynamicsAdaptive(gt_model)
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

    #----------------------------------
    # Setup Adaptive Estimator
    theta_bounds = []
    theta_bounds.append( lambda theta, c1_max=c1_max: theta[0] - c1_max )
    theta_bounds.append( lambda theta, c1_min=c1_min: c1_min - theta[0] )
    theta_bounds.append( lambda theta, c2_max=c2_max: theta[1] - c2_max )
    theta_bounds.append( lambda theta, c2_min=c2_min: c2_min - theta[0] )
    #AE = AdaptiveEstimator(theta_val=[model.c1, model.c2], f=model_adaptive, type='LSE', window=100, Theta=theta_bounds, nx=nx, nu=nu, integrator_type=policy_params['integrator_type'], dT = dt )
    AE = AdaptiveEstimator(theta_val=[model.c1, model.c2], f=model_adaptive, type='ConcurrentLearning', window=100, Theta=theta_bounds, nx=nx, nu=nu, integrator_type='cont', dT=model.dt)

    # ---------------------------------
    # Setup and run simulation

    x0 = np.array(model.x0).reshape((nx, 1))  # 4.0 + model.x0.reshape((nx, 1)) # 23.5*np.ones((nx,1)) #
    dummy_control = DummyControl()
    u_dummy = lambda x,k,u: 0.0

    CL_nom = ClosedLoopSystem(f=model, u=dummy_control, u_nom=u_dummy, dt=model.dt, int_type='cont')

    n_iter = 1
    c1 = model.c1
    c2 = model.c2
    e_nom = []
    e_adapt = []
    conv_cond = []
    torch.manual_seed(seed)
    for kk in range(n_iter):

        # Update model parameters

        c1 = c1_min + (c1_max - c1_min) * torch.rand(1)
        c2 = c2_min + (c2_max - c2_min) * torch.rand(1)
        a = torch.tensor(torch.mm(torch.tensor([[c1], [c2]]), torch.ones(1, Nsim + 1), ),
                         dtype=torch.float32).reshape(nmodel, Nsim + 1)
        CL_nom.f.c1 = c1
        CL_nom.f.c2 = c2
        CL_adapt.f.c1 = c1
        CL_adapt.f.c2 = c2
        CL_adapt.u_nom.exogenous_variables = {'r': torch_ref, 'a': a}

        print('Model parameters: c1 = ' + str(c1) + ', c2 = ' + str(c2))

        # Run simulations
        [X_nom, U_nom] = CL_nom.simulate(x0=x0, N=Nsim)
        [X_adapt, U_adapt] = CL_adapt.simulate(x0=x0, N=Nsim)

        theta_est = []
        Ldk_est = []
        ranks = []
        f_errors = []
        k = 0
        for ii, xi in enumerate(X_nom):

            ui = U_nom[ii,:]

            ae_vals = AE(xi, k, ui)
            theta_est.append( ae_vals[0])
            Ldk_est.append( ae_vals[1] )
            ranki = AE.check_regressor_rank(AE.x, AE.u)
            ranks.append(ranki)

            if ii < Nsim - 1:
                F = AE.compute_parameter_model(xi, ii, ui)

                #f = X_nom[ii + 1, :]
                # -------New try ---------
                f = np.sum( X_nom[ii+1,:] - xi )/dt
                # -------New try ---------
                f_error = np.linalg.norm(F-f)

                f_errors.append(f_error)

                #conv_cond.append( AE.check_convergence_condition(xi, ii, ui) )

            k += 1

        # Compute error
        e_nom.append(np.sum([np.linalg.norm(X_nom[kk, :] - torch_ref.reshape(Nsim + 1, nmodel).numpy()[kk, :]) for kk in
                             range(Nsim + 1)]))
        e_adapt.append(np.sum(
            [np.linalg.norm(X_adapt[kk, :] - torch_ref.reshape(Nsim + 1, nmodel).numpy()[kk, :]) for kk in
             range(Nsim + 1)]))

        if kk < 5:

            plt.figure()
            plt.plot(theta_est, label='theta_est')
            plt.plot([c1 for kk in range(Nsim + 1)], 'k--', label='c1')
            plt.plot([c2 for kk in range(Nsim + 1)], 'r--', label='c2')
            plt.xlabel('Time step')
            plt.ylabel('Values')
            plt.legend()

            plt.figure()
            plt.plot(Ldk_est, label='Ldk_est')
            plt.xlabel('Time step')
            plt.ylabel('Values')
            plt.legend()

            plt.figure()
            plt.plot(ranks, label='rank')
            plt.xlabel('Time step')
            plt.ylabel('Values')
            plt.legend()

            plt.figure()
            plt.plot(f_errors, label='model error')
            plt.xlabel('Time step')
            plt.ylabel('Values')
            plt.legend()

            plt.figure()
            plt.plot(conv_cond, label='conv cond (should be greater than zero')
            plt.xlabel('Time step')
            plt.ylabel('Values')
            plt.legend()

            t = [kk for kk in range(Nsim + 1)]
            plt.figure()
            plt.plot(t, [model.xmax[-1] for kk in range(Nsim + 1)], 'k', linewidth=3.0)
            plt.plot(t, [model.xmin[-1] for kk in range(Nsim + 1)], 'k', linewidth=3.0)
            plt.plot(np_refs, color='navy', label='ref', linewidth=3.0)
            plt.plot(X_nom[:, 0], color='crimson', label='x1 (DPC nominal)', linewidth=2.0)
            plt.plot(X_nom[:, 1], color='chartreuse', label='x2 (DPC nominal)', linewidth=2.0)
            plt.plot(X_adapt[:, 0], color='orange', linestyle='--', label='x1 (DPC adaptive)', linewidth=2.0)
            plt.plot(X_adapt[:, 1], color='magenta', linestyle='--', label='x2 (DPC adaptive)', linewidth=2.0)
            plt.xlabel('Time step')
            plt.ylabel('Tank Level')
            plt.title('c1 = ' + str(c1) + ' , c2 = ' + str(c2))
            plt.legend(loc='lower left')

            plt.figure()
            for ii in range(nu):
                plt.step(range(U_nom.shape[0]), 1.0 * np.ones(U_nom.shape[0]), 'k', linewidth=3.0)
                plt.step(range(U_nom.shape[0]), 0.0 * np.ones(U_nom.shape[0]), 'k', linewidth=3.0)
            plt.plot(U_nom[:, 0], color='crimson', label="u1 (DPC nominal)", linewidth=2.0)
            plt.plot(U_nom[:, 1], color='chartreuse', label="u2 (DPC nominal)", linewidth=2.0)
            plt.plot(U_adapt[:, 0], color='orange', linestyle='--', label="u1 (DPC adaptive)", linewidth=2.0)
            plt.plot(U_adapt[:, 1], color='magenta', linestyle='--', label="u2 (DPC adaptive)", linewidth=2.0)
            plt.title('c1 = ' + str(c1) + ' , c2 = ' + str(c2))
            plt.xlabel('Time step')
            plt.ylabel('Pump and Valve (Inputs)')
            plt.legend(loc='upper right', bbox_to_anchor=(0.77, 0.95))

    plt.figure()
    plt.plot(e_nom, label="nominal")
    plt.plot(e_adapt, label="adaptive")

    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Error')
    plt.legend()

    plt.show()