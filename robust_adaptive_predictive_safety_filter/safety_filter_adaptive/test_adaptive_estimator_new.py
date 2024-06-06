from casadi import *
import matplotlib.pyplot as plt
from neuromancer import psl
import torch
from closed_loop import DPC_control
from closed_loop import DummyControl
from closed_loop import ClosedLoopSystem
import copy
from adaptive_estimator.adaptive_estimator import AdaptiveEstimator

class TwoTankDynamicsAdaptive():
    def __init__(self, model,u=0, W=0):
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
        self.u = u
        self.dt = 1.0
        self.nregressor = 2
        self.nf = 1
        self.nx = 2
        self.nu = 2
        self.ntheta = 2
        self.W = W

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


    def __call__(self, x, k, u, theta):
        '''Forward dynamics of the system based on the model parameters
        x: current state
        k: time
        u: control
        theta: model parameter
        '''

        dhdt = copy.deepcopy(x)
        dhdt[0] = x[0] + self.dt*( theta[0] * (1.0 - u[1]) * u[0] - theta[1]* sqrt(x[0]) )
        dhdt[1] = x[1] + self.dt*( theta[0] * u[1] * u[0] + theta[1] * sqrt(x[0]) - theta[1] * sqrt(x[1]) )

        return dhdt
    def f(self, x, xkp1):
        ''' This is the output function for the adaptive estimator. Note this needs to take into account the discretization used, if any.
               x: current state
               xkp1: next state
               '''

        return (xkp1 - x).T.dot(np.ones((self.nx,1)))/self.dt

    def regressor(self, x, k ,u):
        '''This is the nonlinear component of the dynamics in the form of f = theta^T phi(x), where phi is the regressor, f is the output and theta is the mode parameter matrix
                x: current state
                k: current time
                u: current control
                '''

        phi = np.zeros((self.nregressor, 1))
        phi[0] = (1.0 - u[1]) * u[0] + u[0] * u[1]
        phi[1] = -sqrt(x[0]) + sqrt(x[0]) - sqrt(x[1])

        m = sqrt( 1.0 + phi.T.dot(phi) )
        psi = phi/m

        return psi, m

    def control_regressor(self, phi, x, u):



        phi[0,0] = (1.0 - u[1])*u[0]
        phi[0,1] = -sqrt(x[0])
        phi[1,0] = u[1]*u[0]
        phi[1,1] = sqrt(x[0]) - sqrt(x[1])

        return phi

    def compute_regressor_bound(self):

        # Setup optimization
        opti = Opti()
        x = opti.variable(self.nx, 1)
        u = opti.variable(self.nu, 1)
        f = opti.variable(self.nx, self.ntheta)
        opti.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend

        # Define constraints on states and controls
        f = self.control_regressor(f, x, u)
        for ii in range(self.nx):
            opti.subject_to(x[ii] <= self.xmax[ii])
            opti.subject_to(x[ii] >= self.xmin[ii])
        for ii in range(self.nu):
            opti.subject_to(u[ii] <= self.umax[ii])
            opti.subject_to(u[ii] >= self.umin[ii])

        # Define cost to maximize regressor value, ie., worst case
        ones = np.ones((self.nx, 1))
        fsum = mtimes(ones.T, f).T
        cost = fsum.T@fsum
        opti.minimize(-cost)

        # Solve optimization
        sol = opti.solve()

        # Extract estimate of worst case regressor
        self.phi_max = np.sqrt(sol.value(cost))


    def model_error(self, x, theta_error):
        ''' Computes worst case model error based on current conservative error of model parameters'''

        # 3.0 comes from bounding the regressor matrix using bounds on x and u
        # \| phi \| <= |1-u1| |u0| + |u0||u1| + |sqrt(x1)| <= 3
        return 3.0*self.dt*np.linalg.norm(theta_error)

    def theta_mat(self, theta):
        '''This converts the list/vector form of the model parameters to the matrix form of the model parameters
                theta: list/vector from of model parameters
                '''

        theta_mat = np.zeros((self.nregressor, 1))
        theta_mat[0] = theta[0]
        theta_mat[1] = theta[1]

        return theta_mat

    def theta_demat(self, theta_mat):
        '''This converts the matrix form of the model parameters to the list/vector form
                theta_mat: Matrix form of model parameters'''
        return np.array([theta_mat[0,0], theta_mat[1,0]])




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
        dhdt[0] = c1 * (1.0 - u[1]) * u[0] - c2* sqrt(x[0]) + self.W#*sin(0.2*k)
        dhdt[1] = c1 * u[1] * u[0] + c2 * sqrt(x[0]) - c2 * sqrt(x[1])

        return dhdt


if __name__ == "__main__":



    """
    # # #  Arguments, dimensions, bounds
    """
    Nsim = 2000

    ## perturbation
    W = 0.0

    # ground truth system model
    gt_model = psl.nonautonomous.TwoTank()
    model = TwoTankDynamics(gt_model, W=W)
    model_adaptive = TwoTankDynamicsAdaptive(gt_model, W=W)
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

    #----------------------------
    # Reference trajectory
    r_nsteps = Nsim
    np_refs = psl.signals.step(r_nsteps + 1, 1, min=0.2, max=model.xmax[0], randsteps=int(Nsim/150), rng=np.random.default_rng(seed=seed))
    #np_refs = psl.signals.step(r_nsteps + 1, 1, min=0.2, max=model.xmax[0], randsteps=1, rng=np.random.default_rng(seed=seed))
    #np_refs = psl.signals.periodic(r_nsteps + 1, 1, min=0.2, max=model.xmax[0], periods=int(Nsim/150), form='sin', phase_offset=False, rng=np.random.default_rng(seed=seed))
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

    # Add the control policy to the adaptive model !! Important !!
    model_adaptive.u = u_DPC
    #model_adaptive.compute_regressor_bound()

    # Adaptive
    policy_name = 'two_tank_adaptive'
    version = 4
    policy = torch.load(policy_name + '_policy_' + str(version) + '.pth')
    policy_params = torch.load(policy_name + '_params_' + str(version) + '.pth')
    a = torch.tensor(torch.mm(torch.tensor([[model.c1], [model.c2]]), torch.ones(1, Nsim + 1), ), dtype=torch.float32).reshape(nmodel, Nsim + 1)
    u_ADPC = DPC_control(nu, policy, policy_params, {'r': torch_ref, 'a': a}, umin=model.umin, umax=model.umax)
    dt = policy_params['ts']

    #----------------------------------
    # Setup Adaptive Estimator
    theta_bounds = []
    theta_bounds.append( lambda theta, c1_max=c1_max: theta[0] - c1_max )
    theta_bounds.append( lambda theta, c1_min=c1_min: c1_min - theta[0] )
    theta_bounds.append( lambda theta, c2_max=c2_max: theta[1] - c2_max )
    theta_bounds.append( lambda theta, c2_min=c2_min: c2_min - theta[1] )
    cl_params = {'Gamma': 0.05, 'xi_NG': 0.001, 'xi_CL': 0.3, 'w': W}
    AE = AdaptiveEstimator(theta_val=[model.c1, model.c2], f=model_adaptive, window=100, Theta=theta_bounds, nx=nx, nu=nu, integrator_type=policy_params['integrator_type'], dT=dt, params=cl_params)

    # ---------------------------------
    # Setup and run simulation

    x0 = np.array(model.x0).reshape((nx, 1))  # 4.0 + model.x0.reshape((nx, 1)) # 23.5*np.ones((nx,1)) #
    dummy_control = DummyControl()

    CL_nom = ClosedLoopSystem(f=model, u=dummy_control, u_nom=u_DPC, dt=dt, int_type=policy_params['integrator_type'])
    CL_adapt = ClosedLoopSystem(f=model, u=dummy_control, u_nom=u_ADPC, dt=dt,
                                int_type=policy_params['integrator_type'])

    # Iterate through different model parameters to test
    n_iter = 4
    c1 = model.c1
    c2 = model.c2
    e_nom = []
    e_adapt = []
    conv_cond = []
    seed = 4
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
        f_error_cons = []
        V_est = []
        theta_errors = []
        theta_error_approx = []
        k = 0
        AE.cl_reset()
        # Compute model parameter estimate
        for ii, xi in enumerate(X_nom):

            ui = U_nom[ii,:]

            ae_vals = AE(xi, k, ui)
            theta_est.append( ae_vals[0])
            Ldk_est.append( ae_vals[1] )
            ranki = AE.check_regressor_rank()
            ranks.append(ranki)

            if ii < Nsim - 1:
                # Compute error in dynamics and prediction
                F = AE.compute_parameter_model(xi, k, ui)
                f = model_adaptive(xi, ii, ui, AE.theta_val )
                f_error = np.linalg.norm(X_nom[ii+1,:]-f)
                f_errors.append(f_error)

            f_error_cons.append( model_adaptive.model_error(xi, AE.theta_error) )

            #print('conv cond = '+str(AE.check_convergence_condition()))

            # Compute Lypaunov error and model parameter error
            theta_errors.append( np.linalg.norm( np.array(ae_vals[0]) - np.array([c1, c2]) )**2 )
            theta_error_approx.append( np.linalg.norm( AE.theta_error)**2)
            V_est.append( AE.Gamma*AE.V )

            k += 1

        # Compute tracking error
        e_nom.append(np.sum([np.linalg.norm(X_nom[kk, :] - torch_ref.reshape(Nsim + 1, nmodel).numpy()[kk, :]) for kk in
                             range(Nsim + 1)]))
        e_adapt.append(np.sum(
            [np.linalg.norm(X_adapt[kk, :] - torch_ref.reshape(Nsim + 1, nmodel).numpy()[kk, :]) for kk in
             range(Nsim + 1)]))

        theta_array = np.array(theta_est)

        if kk < 5:

            plt.figure()
            plt.plot(theta_array[:,0], label='c1 estimate')
            plt.plot(theta_array[:, 1], label='c2 estimate')
            plt.plot([c1 for kk in range(Nsim + 1)], 'k--', label='c1')
            plt.plot([c2 for kk in range(Nsim + 1)], 'r--', label='c2')
            plt.xlabel('Time step')
            plt.ylabel('Values')
            plt.legend()

            plt.figure()
            plt.plot(V_est, label='Lyapunov value')
            plt.plot(theta_errors, label='parameter errors')
            plt.plot(theta_error_approx, label='parameter errors approx')
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
            plt.plot(f_error_cons, label='model error bound')
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