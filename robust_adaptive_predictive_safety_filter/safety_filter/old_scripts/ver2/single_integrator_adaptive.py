from casadi import *
import matplotlib.pyplot as plt
import torch
from closed_loop import DummyControl
from closed_loop import ClosedLoopSystem
import copy
from adaptive_estimator.adaptive_estimator import AdaptiveEstimator

class SingleIntegratorAdaptive():
    def __init__(self):

        self.nx = 1
        self.nu = 1
        self.dt = 1

        self.nx = 1
        self.nregressor = 2

        self.c_max = 10.0
        self.c_min = 0.1
        self.x0 = [0.0]

    def f(self, x, xkp1):
        ''' This is the output function for the adaptive estimator. Note this needs to take into account the discretization used, if any.
        x: current state
        xkp1: next state
        '''

        return xkp1

    def regressor(self, x):
        '''This is the nonlinear component of the dynamics in the form of f = theta^T phi(x), where phi is the regressor, f is the output and theta is the mode parameter matrix
        x: current state
        '''
        phi = np.zeros(self.nregressor)
        phi[0] = cos(x)
        phi[1] = 1.0

        m = sqrt( 1.0 + phi.dot(phi) )
        psi = phi/m

        return psi, m

    def theta_mat(self, theta):
        '''This converts the list/vector form of the model parameters to the matrix form of the model parameters
        theta: list/vector from of model parameters
        '''

        theta_mat = np.zeros((2,1))
        theta_mat[0] = theta[0]
        theta_mat[1] = theta[1]
        return theta_mat

    def theta_demat(self, theta_mat):
        '''This converts the matrix form of the model parameters to the list/vector form
        theta_mat: Matrix form of model parameters'''

        return theta_mat.flatten()


class SingleIntegratorDynamics():
    def __init__(self):

        self.nx = 1
        self.nu = 1
        self.c = 2.0
        self.d = 0.5
        self.c_max = 10.0
        self.c_min = 0.1
        self.dt = 1
        self.x0 = [0.0]

    def __call__(self, x, k, u):

        dhdt = copy.deepcopy(x)
        dhdt[0] = self.c*cos(x) + self.d
        return dhdt


if __name__ == "__main__":



    """
    # # #  Arguments, dimensions, bounds
    """
    Nsim = 2000

    # ground truth system model
    model = SingleIntegratorDynamics()
    model_adaptive = SingleIntegratorAdaptive()
    nx = model.nx
    nu = model.nu
    nmodel = 2

    # Model bounds
    c_max = model.c_max
    c_min = model.c_min
    model_max = torch.tensor([[c_max, c_max]])

    seed = 4
    torch.manual_seed(seed)


    #----------------------------------
    # Setup Adaptive Estimator
    theta_bounds = []
    theta_bounds.append( lambda theta, c_max=c_max: theta[0] - c_max )
    theta_bounds.append( lambda theta, c_min=c_min: c_min - theta[0] )
    theta_bounds.append( lambda theta, c_max=c_max: theta[1] - c_max )
    theta_bounds.append(lambda theta, c_min=c_min: c_min - theta[1])
    #AE = AdaptiveEstimator(theta_val=[model.c_min], f=model_adaptive, type='LSE', window=100, Theta=theta_bounds, nx=nx, nu=nu, integrator_type='cont', dT=model.dt)
    AE = AdaptiveEstimator(theta_val=[model.c_min, model.c_min], f=model_adaptive, type='ConcurrentLearning', window=100, Theta=theta_bounds, nx=nx, nu=nu, integrator_type='cont', dT=model.dt)

    # ---------------------------------
    # Setup and run simulation

    x0 = np.array(model.x0).reshape((nx, 1))  # 4.0 + model.x0.reshape((nx, 1)) # 23.5*np.ones((nx,1)) #
    dummy_control = DummyControl()
    u_nom = lambda x, k: 0.0
    CL_nom = ClosedLoopSystem(f=model, u=dummy_control, u_nom=u_nom, dt=model.dt, int_type='cont')

    n_iter = 1
    c = model.c
    d = model.d
    e_nom = []
    e_adapt = []
    conv_cond = []

    # Update model parameters

    c = model.c
    d = model.d
    print('Model parameters: c = ' + str(c) )
    print('Model parameters: d = ' + str(d))

    # Run simulations
    [X_nom, U_nom] = CL_nom.simulate(x0=x0, N=Nsim)

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
        ranki = AE.check_regressor_rank()
        ranks.append(ranki)

        if ii < Nsim - 1:
            F = AE.compute_parameter_model(xi)
            f = X_nom[ii + 1, :]

            f_error = np.linalg.norm(F-f)

            f_errors.append(f_error)


    plt.figure()
    plt.plot(theta_est, label='theta_est')
    plt.plot([c for kk in range(Nsim + 1)], 'k--', label='c')
    plt.plot([d for kk in range(Nsim + 1)], 'b--', label='d')
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
    plt.plot(X_nom[:, 0], color='crimson', label='x', linewidth=2.0)
    plt.xlabel('Time step')
    plt.ylabel('X')
    plt.title('c = ' + str(c) )
    plt.legend(loc='lower left')

    plt.figure()
    plt.plot(U_nom[:, 0], color='crimson', label="u1 (DPC nominal)", linewidth=2.0)
    plt.title('c = ' + str(c) )
    plt.xlabel('Time step')
    plt.ylabel('U')
    plt.legend(loc='upper right', bbox_to_anchor=(0.77, 0.95))


    plt.show()