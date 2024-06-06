from casadi import *
import matplotlib.pyplot as plt
from neuromancer import psl
import torch
from closed_loop import DPC_control, DummyNominalControl, DummyControl
from predictive_safety_filter import adaptive_safety_filter as sf
from closed_loop import ClosedLoopSystem, ClosedLoopSystemAdaptive
import copy
from adaptive_estimator.adaptive_estimator import AdaptiveEstimator
from two_tank_barrier_test import TestBarrierFunction, ValidControl


class TwoTankDynamicsAdaptive():
    def __init__(self, model, W=0):
        self.model = model
        self.c1 = model.params[2]['c1']  # inlet valve coefficient
        self.c2 = model.params[2]['c2']   # tank outlet coefficient
        self.c1_max = 0.081
        self.c1_min = 0.079
        self.c2_max = 0.041
        self.c2_min = 0.039
        self.theta_max = [self.c1_max, self.c2_max]
        self.theta_min = [self.c1_min, self.c2_min]
        self.umax = np.array([1., 1.])
        self.umin = np.array([0, 0])
        self.xmax = [1., 1.]
        self.xmin = [0.2, 0.2]
        self.x0 = [0.3, 0.4]
        self.dt = 1.0
        self.nregressor = 2
        self.nf = 1
        self.nx = 2
        self.nu = 2
        self.ntheta = 2
        self.W = W
        z = [0.96179541, -4.68358656, 9.79316323, -11.55994252, 8.58609308, -4.3743985, 2.13463916, 0.14225085]
        self.sqrt_approx = lambda x, z=z: polyval(z, x)

        # Define casadi optimization parameters
        self.p_opts = {"expand": True, "print_time": False, "verbose": False}
        self.s_opts = {'max_iter': 1000,
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


    def __call__(self, x, k, u, theta):
        '''Forward dynamics of the system based on the model parameters
        x: current state
        k: time
        u: control
        theta: model parameter
        '''

        dhdt = copy.deepcopy(x)
        #dhdt[0] = x[0] + self.dt*( theta[0] * (1.0 - u[1]) * u[0] - theta[1]* sqrt(x[0]) )
        #dhdt[1] = x[1] + self.dt*( theta[0] * u[1] * u[0] + theta[1] * sqrt(x[0]) - theta[1] * sqrt(x[1]) )
        dhdt[0] = x[0] + self.dt * (theta[0] * (1.0 - u[1]) * u[0] - theta[1] * self.sqrt_approx(x[0]))
        dhdt[1] = x[1] + self.dt * (theta[0] * u[1] * u[0] + theta[1] * self.sqrt_approx(x[0]) - theta[1] * self.sqrt_approx(x[1]))

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
        phi[1] = -self.sqrt_approx(x[0]) + self.sqrt_approx(x[0]) - self.sqrt_approx(x[1])
        #phi[0] = (1.0 - u[1]) * u[0] + u[0] * u[1]
        #phi[1] = -sqrt(x[0]) + sqrt(x[0]) - sqrt(x[1])

        m = sqrt( 1.0 + phi.T.dot(phi) )
        psi = phi/m

        return psi, m

    def control_regressor(self, phi, x, u):



        #phi[0,0] = (1.0 - u[1])*u[0]
        #phi[0,1] = -sqrt(x[0])
        #phi[1,0] = u[1]*u[0]
        #phi[1,1] = sqrt(x[0]) - sqrt(x[1])

        phi[0, 0] = (1.0 - u[1]) * u[0]
        phi[0, 1] = -self.sqrt_approx(x[0])
        phi[1, 0] = u[1] * u[0]
        phi[1, 1] = self.sqrt_approx(x[0]) - self.sqrt_approx(x[1])

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


    def model_error_old(self, x, theta_error):
        ''' Computes worst case model error based on current conservative error of model parameters'''

        # 3.0 comes from bounding the regressor matrix using bounds on x and u
        # \| phi \| <= |1-u1| |u0| + |u0||u1| + |sqrt(x1)| <= 3
        return 3.0*self.dt*np.linalg.norm(theta_error)

    def model_error(self, x, theta_error):
        ''' Computes worst case model error based on current conservative error of model parameters'''

        # Setup optimization
        opti = Opti()
        x = opti.variable(self.nx, 1)
        u = opti.variable(self.nu, 1)
        #theta = opti.variable(self.ntheta,1)
        #theta_true = opti.variable(self.ntheta, 1)
        phi = opti.variable(self.nx, self.ntheta)
        #f = self.__call__( x, 0, u, theta)
        #f_true = self.__call__(x, 0, u, theta_true)
        reg = self.control_regressor(phi, x, u)
        opti.solver("ipopt", self.p_opts, self.s_opts)  # set numerical backend

        # Define constraints on states and controls
        for ii in range(self.nx):
            opti.subject_to(x[ii] <= self.xmax[ii])
            opti.subject_to(x[ii] >= self.xmin[ii])
        for ii in range(self.nu):
            opti.subject_to(u[ii] <= self.umax[ii])
            opti.subject_to(u[ii] >= self.umin[ii])
        # for ii in range(self.ntheta):
        #     opti.subject_to(theta[ii] <= self.theta_max[ii])
        #     opti.subject_to(theta[ii] >= self.theta_min[ii])
        #     opti.subject_to(theta_true[ii] <= self.theta_max[ii])
        #     opti.subject_to(theta_true[ii] >= self.theta_min[ii])

        # Define cost to maximize regressor value, ie., worst case
        reg_square = reg.T@reg
        cost = reg_square[0,0] + reg_square[1,1]
        #cost = (f-f_true).T@(f- f_true)
        opti.minimize(-cost)

        # Solve optimization
        sol = opti.solve()

        # Extract estimate of worst case regressor
        return np.sqrt(sol.value(cost))*theta_error
        #return np.sqrt(sol.value(cost)) #

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
    def __init__(self, model,W=0, dt=1.0):
        self.model = model
        self.c1 = model.params[2]['c1']  # inlet valve coefficient
        self.c2 = model.params[2]['c2']   # tank outlet coefficient
        self.umax = np.array([1, 1])
        self.umin = np.array([0, 0])
        self.xmax = [1., 1.]
        self.xmin = [0.2, 0.2]
        self.x0 = [0.3, 0.4]
        self.W = W
        self.dt = dt
        z = [0.96179541, -4.68358656, 9.79316323, -11.55994252, 8.58609308, -4.3743985, 2.13463916, 0.14225085]
        self.sqrt_approx = lambda x, z=z: polyval(z, x)

    def __call__(self, x, k, u):

        dhdt = copy.deepcopy(x)
        dhdt[0] = x[0] + self.dt * ( c1 * (1.0 - u[1]) * u[0] - c2 * self.sqrt_approx(x[0]) )
        dhdt[1] = x[1] + self.dt * ( c1 * u[1] * u[0] + c2 * self.sqrt_approx(x[0]) - c2 * self.sqrt_approx(x[1]) )

        return dhdt


if __name__ == "__main__":



    """
    # # #  Arguments, dimensions, bounds
    """


    TESTBARRIER = False
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
    c1_max = model_adaptive.c1_max
    c1_min = model_adaptive.c1_min
    c2_max = model_adaptive.c2_max
    c2_min = model_adaptive.c2_min
    model_max = torch.tensor([[c1_max, c2_max]])
    model_min = torch.tensor([[c1_min, c2_min]])

    seed = 4

    #----------------------------
    # Reference trajectory
    r_nsteps = Nsim
    rmin = model.xmin[0] #0.45 #
    rmax = model.xmax[0] #0.7 #
    #np_refs = psl.signals.step(r_nsteps + 1, 1, min=rmin, max=rmax, randsteps=int(Nsim/150), rng=np.random.default_rng(seed=seed))
    #np_refs = psl.signals.step(r_nsteps + 1, 1, min=0.2, max=model.xmax[0], randsteps=1, rng=np.random.default_rng(seed=seed))
    #np_refs = psl.signals.periodic(r_nsteps + 1, 1, min=rmin, max=rmax, periods=int(Nsim/150), form='sin', phase_offset=False, rng=np.random.default_rng(seed=seed))
    np_refs = psl.signals.sines(r_nsteps + 1, 1, min=rmin, max=rmax, periods=int(Nsim/150),nwaves=4, form='sin', rng=np.random.default_rng(seed=seed))
    R = torch.tensor(np_refs, dtype=torch.float32).reshape(  1, r_nsteps + 1)
    torch_ref = torch.cat([R, R], dim=0)

    #-------------------------------
    # Load Adaptive DPC policy
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
    initial_theta_bound = {'theta1': np.array([c1_max, c2_max]), 'theta2': np.array([c1_min, c2_min])}
    print('c1 max = '+str(c1_max))
    print('c1 min = ' + str(c1_min))
    print('c2 max = ' + str(c2_max))
    print('c2 min = ' + str(c2_min))
    cl_params = {'Gamma': 0.05, 'xi_NG': 0.8, 'xi_CL': 0.3, 'w': W}
    Ta = 1 # Sampling time for adaptive estimation
    AE = AdaptiveEstimator(theta_val=[model.c1, model.c2], f=model_adaptive, window=105, Theta=theta_bounds, nx=nx, nu=nu, integrator_type=policy_params['integrator_type'], dT=dt, params=cl_params, sampling_time=Ta, initial_theta_bound=initial_theta_bound)

    # Get initial worst-case error bound
    Ld0 = model_adaptive.model_error(0, AE.theta_error)
    print('Ld0 = '+str(Ld0))

    print('theta error = '+str(AE.theta_error))
    d_bound = model_adaptive.model_error(0,AE.theta_error)
    print('d bound = '+str(d_bound))

    #-----------------------------------
    # Setup Safety Filter

    # Define parameters for the safety filter
    params = {}
    params['terminal_conditions'] = []
    params['dt'] = policy_params['ts']  # sampling time
    params['delta'] = 0.00  # Small robustness margin for constraint tightening
    params['alpha1='] = 1000000.0  # Multiplication factor for feasibility parameters
    params['alpha2'] = 1.0  # Multiplication factor for feasibility parameters
    params['integrator_type'] = 'cont' #policy_params['integrator_type']  # Integrator type for the safety filter
    ebar = 0.00002835
    #dbar = 0.001
    #wbar = ebar + dbar

    # -----------------------------
    # Define system constraints
    terminal_constraint = {}
    state_constraints = {}
    input_constraints = {}

    # -----------------------------
    # Define system constraints
    terminal_constraint = {}
    state_constraints = {}
    input_constraints = {}
    params['nx'] = nx
    params['nu'] = nu
    params['ny'] = ny
    umax = np.array(
        [1.0, 1.0])  # model.umax.flatten()  # *np.ones((nu,1)) #[model.umax]  # Input constraint parameters(max)
    umin = np.array(
        [0, 0])  # model.umin.flatten()  # model.umin*np.ones((nu,1)) #[model.umin] # Input constraint parameters (min)

    # State constraint functions: g(x) <= 0
    kk = 0
    for ii in range(nx):
        state_constraints[kk] = lambda x, k, xmax=model.xmax, ind=ii: x[ind] - xmax[ind]
        kk += 1
        state_constraints[kk] = lambda x, k, xmin=model.xmin, ind=ii: xmin[ind] - x[ind]
        kk += 1

    # Terminal constraint set
    xr_setpoint = np.array([0.6, 0.6]).reshape((nx, 1))
    h_eps = 0.14 #0.12
    p2 = 2.69
    P = np.array([[1.0, 0.0], [0.0, p2]])
    hf = lambda x, k, xr=xr_setpoint, h_eps=h_eps: (x - xr_setpoint).T @ P @ (x - xr_setpoint) - h_eps

    # Find the level sets of the terminal constraints
    hf_level_plus = lambda a: xr_setpoint[-1] + sqrt(h_eps + a)
    hf_level_minus = lambda a: xr_setpoint[-1] - sqrt(h_eps + a)

    # Input constraints: h(u) <= 0
    kk = 0
    for ii in range(nu):
        input_constraints[kk] = lambda u, ii=ii, umax=umax: u[ii] - umax[ii]
        kk += 1
        input_constraints[kk] = lambda u, ii=ii, umin=umin: umin[ii] - u[ii]
        kk += 1

    constraints = {}
    constraints['state_constraints'] = state_constraints
    constraints['hf'] = hf
    constraints['input_constraints'] = input_constraints

    # Robustness margins (Need to double check that everything is w.r.t the same norm)
    Lf_x = 1.0 + model_adaptive.c2_max * dt * sqrt((3.0 + sqrt(5.0)) / model.xmin[-1])
    Lh_x = 1.0
    Lf_theta = dt*(1 + sqrt(1.0)) # Computed by bounding \phi on u1,2 \in [0,1], x1,2 \in [0.2, 1]
    #Ld = sqrt(2.0) * wbar
    Lhf_x_func = lambda eps_var, p2=p2: 2.0 * sqrt((1 + p2) * eps_var)
    Lhf_x = Lhf_x_func(eps_var=h_eps)

    params['terminal_conditions'].append({'N': 2, 'Ld': d_bound})
    params['terminal_conditions'].append({'N': 6, 'Ld': 0.00145})
    params['terminal_conditions'].append({'N': 20, 'Ld': 0.0000542})
    params['terminal_conditions'].append({'N': 30, 'Ld': 0.00002})
    params['robust_margins'] = {'Lh_x': Lh_x,  'Lf_x': Lf_x, 'Lhf_x': Lhf_x}  # Robustness margin for handling perturbations

    Nmax = max([params['terminal_conditions'][ii]['N'] for ii in range(len(params['terminal_conditions']))])
    R_total = {'N': [], 'Ld': [], 'R': []}
    for ii in range(len(params['terminal_conditions'])):
        for jj in range(len(params['terminal_conditions'])):
            Ldi = params['terminal_conditions'][ii]['Ld']
            Nj = params['terminal_conditions'][jj]['N']
            R_total['N'].append( Nj )
            R_total['Ld'].append( Ldi )
            R_total['R'].append( Ldi*(Lf_x)**(Nj-1) )
    Rmin = min([ Ri for Ri in R_total['R'] ])
    T = []
    for ii in range(len(params['terminal_conditions'])):
        Ldi = params['terminal_conditions'][ii]['Ld']
        Ni = params['terminal_conditions'][ii]['N']
        T.append( Ldi*(Lf_x)**(Ni-1))
    Tmax_index = T.index(max(T))
    print('max T = '+str(T[Tmax_index]))
    print('max T, N= '+str(params['terminal_conditions'][Tmax_index]['N']))
    print('max Ld = ' + str(params['terminal_conditions'][Tmax_index]['Ld']))


    options = {}
    options['use_feas'] = False
    options['time-varying'] = False
    options['event-trigger'] = True
    options_nonrob = {}
    options_nonrob['time-varying'] = False
    options_nonrob['event-trigger'] = False

    # Set up the safety filter and nominal control
    x0 = np.array(model.x0).reshape((nx, 1))  # 4.0 + model.x0.reshape((nx, 1)) # 23.5*np.ones((nx,1)) #
    u_sf = sf.AdaptiveSafetyFilter(x0=x0, theta0=[model.c1, model.c2], f=model_adaptive, params=params, constraints=constraints, options=options)


    #-----------------------------------
    # Test barrier function conditions
    if TESTBARRIER:
        valid_u = ValidControl(dt, xr_setpoint, p2)
        barrier_test = TestBarrierFunction(model_adaptive, valid_u, dt, params['robust_margins'], hf, integrator_type=params['integrator_type'])
        for ii in range(len(params['terminal_conditions'])):
            barrier_test.check_constraint_conditions(params['terminal_conditions'][ii]['N'], params['terminal_conditions'][ii]['Ld'], Nsim, hf_level_plus, hf_level_minus)
        x_range = np.linspace(xr_setpoint[-1] - sqrt(h_eps), xr_setpoint[-1] + sqrt(h_eps), 100)
        theta1_range = np.linspace(model_adaptive.c1_min, model_adaptive.c1_max, 10)
        theta2_range = np.linspace(model_adaptive.c2_min, model_adaptive.c2_max, 10)
        c1_dummy = model_adaptive.c1_min #(model_adaptive.c1_max + model_adaptive.c1_min)/2.0
        c2_dummy = model_adaptive.c2_min #(model_adaptive.c2_max + model_adaptive.c2_min)/2.0
        barrier_test.eval(params['terminal_conditions'][Tmax_index]['N'], params['terminal_conditions'][Tmax_index]['Ld'],x_range, x_range, theta1_range, theta2_range)
        #barrier_test.eval(params['terminal_conditions'][Tmax_index]['N'], params['terminal_conditions'][Tmax_index]['Ld'], x_range, x_range, [c1_dummy], [c2_dummy], Lf_theta=Lf_theta, thetabound=AE.theta_error)

    # ---------------------------------
    # Setup and run simulation
    #u_rand = lambda x,k: np.random.rand(2,1)
    #u_nom = DummyNominalControl(unom=u_rand)
    dummy_control = DummyControl()
    #CL_adapt = ClosedLoopSystemAdaptive(f=model, u=u_sf, u_nom=u_nom, adaptive_estimator=AE, dt=dt, int_type='cont')
    #CL_adapt = ClosedLoopSystemAdaptive(f=model, u=dummy_control, u_nom=u_ADPC, adaptive_estimator=AE, dt=dt, int_type='cont')
    CL_adapt = ClosedLoopSystemAdaptive(f=model, u=u_sf, u_nom=u_ADPC, adaptive_estimator=AE, dt=dt, int_type='cont')

    # Iterate through different model parameters to test
    n_iter = 1
    c1 = model.c1
    c2 = model.c2
    e_adapt = []
    conv_cond = []
    seed = 4
    torch.manual_seed(seed)
    for kk in range(n_iter):

        # Update model parameters

        c1 = c1_min + (c1_max - c1_min) * torch.rand(1).detach().numpy()
        c2 = c2_min + (c2_max - c2_min) * torch.rand(1).detach().numpy()
        CL_adapt.f.c1 = c1
        CL_adapt.f.c2 = c2
        print('Model parameters: c1 = ' + str(c1) + ', c2 = ' + str(c2))


        # Run simulations
        [X_adapt, U_adapt, Theta_adapt] = CL_adapt.simulate(x0=x0, N=Nsim)


        # Compute tracking error
        e_adapt.append(np.sum(
            [np.linalg.norm(X_adapt[kk, :] - torch_ref.reshape(Nsim + 1, nmodel).numpy()[kk, :]) for kk in
             range(Nsim + 1)]))

        if kk < 5:

            t = [kk for kk in range(Nsim + 1)]
            plt.figure()
            plt.plot(Theta_adapt[:, 0], color='blue', label='theta 1', linewidth=2.0)
            plt.plot(Theta_adapt[:, 1], color='red', label='theta 2 ', linewidth=2.0)
            plt.plot([c1 for kk in range(Nsim + 1)], 'b--', label='c1')
            plt.plot([c2 for kk in range(Nsim + 1)], 'r--', label='c2')
            plt.xlabel('Time step')
            plt.ylabel('Model parameter estimates')
            plt.title('c1 = ' + str(c1) + ' , c2 = ' + str(c2))
            plt.legend(loc='lower left')

            t = [kk for kk in range(Nsim + 1)]
            plt.figure()
            plt.plot(t, [model.xmax[-1] for kk in range(Nsim + 1)], 'k', linewidth=3.0)
            plt.plot(t, [model.xmin[-1] for kk in range(Nsim + 1)], 'k', linewidth=3.0)
            plt.plot(np_refs, color='navy', label='ref', linewidth=3.0)
            plt.plot(X_adapt[:, 0], color='orange', linestyle='--', label='x1 (DPC adaptive)', linewidth=2.0)
            plt.plot(X_adapt[:, 1], color='magenta', linestyle='--', label='x2 (DPC adaptive)', linewidth=2.0)
            plt.xlabel('Time step')
            plt.ylabel('Tank Level')
            plt.title('c1 = ' + str(c1) + ' , c2 = ' + str(c2))
            plt.legend(loc='lower left')

            plt.figure()
            for ii in range(nu):
                plt.step(range(U_adapt.shape[0]), 1.0 * np.ones(U_adapt.shape[0]), 'k', linewidth=3.0)
                plt.step(range(U_adapt.shape[0]), 0.0 * np.ones(U_adapt.shape[0]), 'k', linewidth=3.0)
            plt.plot(U_adapt[:, 0], color='orange', linestyle='--', label="u1 (DPC adaptive)", linewidth=2.0)
            plt.plot(U_adapt[:, 1], color='magenta', linestyle='--', label="u2 (DPC adaptive)", linewidth=2.0)
            plt.title('c1 = ' + str(c1) + ' , c2 = ' + str(c2))
            plt.xlabel('Time step')
            plt.ylabel('Pump and Valve (Inputs)')
            plt.legend(loc='upper right', bbox_to_anchor=(0.77, 0.95))

    plt.figure()
    plt.plot(e_adapt, label="adaptive")

    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Error')
    plt.legend()

    plt.show()