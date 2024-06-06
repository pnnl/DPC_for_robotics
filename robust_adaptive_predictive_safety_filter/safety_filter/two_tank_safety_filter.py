from casadi import *
from predictive_safety_filter import safety_filter as sf
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from neuromancer import psl
import torch
from closed_loop import DPC_control
from closed_loop import DummyControl
from closed_loop import ClosedLoopSystem
from predictive_safety_filter.safety_filter import integrate
import copy
import time


class TestBarrierFunction():

    def __init__(self, model, u, dt, rob_marg, hf):
        self.model = model
        self.u = u
        self.dt = dt
        self.hf = hf
        self.rob_marg = rob_marg

    def eval(self, x1_range, x2_range):
        '''Sample points from x1_range and x2_range and check if a control input is available to satisfy the hf condition'''


        print('Checking barrier function...')
        print('rob marg = '+str(self.rob_marg))
        print('c1 = '+str(model.c1))
        print('c2 = '+str(model.c2))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        figh = plt.figure()
        axh = figh.add_subplot(111, projection='3d')
        max_hf_check =- np.inf

        for x1 in x1_range:
            for x2 in x2_range:

                bool_val = False
                x = np.array([x1, x2])

                # Only perform the check if we are in the sublevel set of hf
                if hf(x, 0) <= 0:

                    # Check if a zero control satisfies the conditions
                    f_x = integrate(self.model, x, 0, np.array([0, 0]), dt, integrator_type='Euler')
                    if hf(f_x, 0) <= -self.rob_marg:
                        ax.scatter(x1, x2, 0, color='b')
                        axh.scatter(x1, x2, hf(f_x, 0) + self.rob_marg, color='b')

                    # If a zero control does not satisfy the condition, compute the optimal control
                    else:
                        u = self.u(x, 0)
                        u1 = u[0]
                        u2 = u[1]

                        # Check if the optimal control satisfies the hf condition
                        f_x = integrate(self.model, x, 0, np.array([u1, u2]), dt, integrator_type='Euler')
                        hf_check = hf(f_x, 0) + self.rob_marg
                        ax.scatter(x1, x2, u1, color='r', )
                        ax.scatter(x1, x2, u2, color='g')
                        axh.scatter(x1, x2, hf_check, color='m')
                        max_hf_check = max([max_hf_check, hf_check])

                        # If the condition is not satisfies, print the violating points
                        if hf_check > 0.0:
                            print('hf violated at x1 = ' + str(x1) + ', x2 = ' + str(x2))
                            print('hf_check = ' + str(hf_check))
                            print('bool val ' + str(bool_val))

        print('max hf_check = '+str(max_hf_check))
        print('Check complete.')
        # Plot the results for verification
        plt.figure(fig)
        plt.xlabel('x1')
        plt.ylabel('x2')
        ax.set_zlabel('u')
        plt.figure(figh)
        plt.xlabel('x1')
        plt.ylabel('x2')
        axh.set_zlabel('hf + Lhx Ld (Lfx)^N-1')
        plt.show()

class ValidControl():

    def __init__(self, c1, c2, dt, r, rho):
        self.c1 = c1
        self.c2 = c2
        self.dt = dt
        self.r = r
        self.rho = rho

        self.opti = Opti()
        self.y1 = self.opti.parameter()
        self.y2 = self.opti.parameter()
        self.u1 = self.opti.variable()
        self.u2 = self.opti.variable()

        self.opti.subject_to(self.u1 >= 0.0)
        self.opti.subject_to(self.u2 >= 0.0)
        self.opti.subject_to(self.u1 <= 1.0)
        self.opti.subject_to(self.u2 <= 1.0)
        self.opti.minimize((self.c1 * (1 - self.u2) * self.u1 - self.y1) ** 2 + self.rho * (self.c1 * self.u1 * self.u2 - self.y2) ** 2)
        self.p_opts = {"expand": True, "print_time": False, "verbose": False}
        self.s_opts = {"print_level": 1}
        self.opti.solver("ipopt", self.p_opts, self.s_opts)
    def __call__(self, x, k):

        x1 = x[0]
        x2 = x[1]
        r1 = self.r[0]
        r2 = self.r[1]

        y1 = -(x1 - r1) / self.dt + self.c2 * sqrt(x1)
        y2 = -(x2 - r2) / self.dt - self.c2 * sqrt(x1) + self.c2 * sqrt(x2)

        self.opti.set_value(self.y1, y1)
        self.opti.set_value(self.y2, y2)
        sol = self.opti.solve()
        u1 = sol.value(self.u1)
        u2 = sol.value(self.u2)

        u = np.array([u1, u2])

        #u1 = ( -x1 - x2 + r1 + r2 + self.dt*self.c2*self.alpha*tanh(x2) )/( self.dt*self.c1 )
        #u2 = 1.0 + ( x1 - self.dt*self.c2*tanh(x1) - r1 )/( -x1 - x2 + r1 + r2 + self.dt*self.c2*tanh(x2) ) #( x1 + self.dt*( self.c1*u1 - self.c2*self.alpha*tanh(x1) ) - r1)/( self.c1*u1*self.dt ) #

        #u = np.array([u1,u2])

        return u

class TwoTankDynamics():
    def __init__(self, model):
        self.model = model
        self.c1 = model.params[2]['c1']  # inlet valve coefficient
        self.c2 = model.params[2]['c2']   # tank outlet coefficient
        self.umax = np.array([1, 1])
        self.umin = np.array([0, 0])
        self.wbar = 0.1
        self.xmax = [1.0, 1.0]
        self.xmin = [0.2, 0.2]
        self.alpha = 0.98  # 0.0  #
        self.beta = 0.25
        self.x0 = [0.5, 0.5]
        z = [  0.96179541,  -4.68358656,   9.79316323, -11.55994252,   8.58609308,  -4.3743985,    2.13463916,   0.14225085]
        self.sqrt_approx = lambda x, z=z: polyval(z, x)

    def __call__(self, x, k, u):

        dhdt = copy.deepcopy(x)
        dhdt[0] = self.c1 * (1.0 - u[1]) * u[0] - self.c2 * self.sqrt_approx(x[0])
        dhdt[1] = self.c1 * u[1] * u[0] + self.c2 * self.sqrt_approx(x[0]) - self.c2 * self.sqrt_approx(x[1])


        #return -x + u
        return dhdt

class TwoTankDynamicsTrue():
    def __init__(self, model,dbar=0):
        self.model = model
        self.c1 = model.params[2]['c1']  # inlet valve coefficient
        self.c2 = model.params[2]['c2']   # tank outlet coefficient
        self.umax = np.array([1, 1])
        self.umin = np.array([0, 0])
        self.xmax = [1, 1]
        self.xmin = [0, 0]
        self.x0 = [0.1, 0.1]
        self.w = lambda k, dbar=dbar: dbar*sin(k)

    def __call__(self, x, k, u):

        dhdt = copy.deepcopy(x)
        dhdt[0] = self.c1 * (1.0 - u[1]) * u[0] - self.c2 * sqrt(x[0]) + self.w(k)
        dhdt[1] = self.c1 * u[1] * u[0] + self.c2 * sqrt(x[0]) - self.c2 * sqrt(x[1]) + self.w(k)

        return dhdt


if __name__ == "__main__":




    """
    # # #  Arguments, dimensions, bounds
    """
    Nsim = 750

    ROBUSTNESS_TEST = True #False # Compare with non-robust safety filter

    # ground truth system model
    gt_model = psl.nonautonomous.TwoTank()
    model = TwoTankDynamics(gt_model)
    nx = gt_model.nx
    ny = gt_model.nx
    nu = gt_model.nu
    ebar = 0.00002835
    if ROBUSTNESS_TEST:
        dbar = 0.001
    else:
        dbar = 0.00001 #   FOR COMPARISON, d= 0.001, N = 6, No robust margins
    wbar = ebar+dbar
    seed = 4

    # Construct the polynomial approximation of sqrt()
    # xf = np.linspace(model.xmin[-1], model.xmax[-1], 1000)
    # yf = np.array([sqrt(xi) for xi in xf])
    # z = np.polyfit(xf, yf, 7)
    # print(z)
    # emax = max([abs(polyval(z,xi) -sqrt(xi)) for xi in xf  ])
    # print('max poly error = '+str(emax))
    # raise (SystemExit)

    true_model = TwoTankDynamicsTrue(gt_model, dbar=dbar)

    #----------------------------
    # Reference trajectory
    r_nsteps = Nsim
    np_refs = psl.signals.step(r_nsteps + 1, 1, min=model.xmin[0], max=model.xmax[0], randsteps=5, rng=np.random.default_rng(seed=seed))
    R = torch.tensor(np_refs, dtype=torch.float32).reshape(  1, r_nsteps + 1)
    torch_ref = torch.cat([R, R], dim=0)

    #-------------------------------
    # Load DPC policy
    policy_name = 'two_tank'
    version = 4
    policy = torch.load(policy_name+'_policy_'+str(version)+'.pth')
    policy_params = torch.load(policy_name + '_params_'+str(version)+'.pth')
    u_DPC = DPC_control(nu, policy, policy_params, {'r': torch_ref}, umin=model.umin, umax=model.umax)
    dt = policy_params['ts']

    # -----------------------------
    # Define parameters
    # -----------------------------
    # Define parameters for the safety filter
    params = {}
    if ROBUSTNESS_TEST:
        params['N'] = 6  #  prediction horizon for robustness test
    else:
        params['N'] = 20 #  prediction horizon for nonrobustness test
    params['dt'] = policy_params['ts']  # sampling time
    params['delta'] = 0.00000005  # Small robustness margin for constraint tightening
    params['alpha1='] = 1000000.0  # Multiplication factor for feasibility parameters
    params['alpha2'] = 1.0  # Multiplication factor for feasibility parameters
    params['integrator_type'] = policy_params['integrator_type']  # Integrator type for the safety filter

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
    umax = np.array([1.0, 1.0]) #model.umax.flatten()  # *np.ones((nu,1)) #[model.umax]  # Input constraint parameters(max)
    umin = np.array([0, 0]) #model.umin.flatten()  # model.umin*np.ones((nu,1)) #[model.umin] # Input constraint parameters (min)

    # State constraint functions: g(x) <= 0
    kk = 0
    for ii in range(nx):
        state_constraints[kk] = lambda x, k, xmax=model.xmax, ind=ii: x[ind] - xmax[ind]
        kk += 1
        state_constraints[kk] = lambda x, k, xmin=model.xmin, ind=ii: xmin[ind] - x[ind]
        kk += 1

    # Terminal constraint set
    xr_setpoint = np.array([0.63, 0.63]).reshape((nx,1)) #[0.5, 0.5] #0.5 * (model.xmax[-1] + model.xmin[-1])*np.ones((nx,1)) #[0.04, 0.02] #
    h_eps = 0.12 #0.08 #0.1
    p2 = 2.69 #2.0
    P = np.array([[1.0, 0.0],[0.0, p2]])
    hf = lambda x, k, xr=xr_setpoint, h_eps=h_eps: (x - xr_setpoint).T @P@ (x - xr_setpoint) - h_eps

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



    # Robustness margins
    Lf_x = 1 + model.c2*dt*sqrt( (3.0 + sqrt(5.0))/model.xmin[-1] )
    Lh_x = 1.0
    Ld = sqrt(2.0)*wbar

    # Checking barrier conditions
    Lhf_x_func = lambda eps_var, p2=p2: 2.0*sqrt( (1+p2) * eps_var ) #2.0*sqrt(eps_var)


    plt.figure()
    xr_range = np.linspace(0, 1, 50)
    plt.plot(xr_range,[Lhf_x_func(eps_var=h_eps)*Ld*Lf_x**(params['N']-1) for xri in xr_range], label='Lhfx*Ld*Lfx^(N-1)')
    plt.plot(xr_range, [Ld for xri in xr_range], label='Ld' )
    plt.plot(xr_range, [h_eps for xr in xr_range], label='eps_var')
    plt.axvline(xr_setpoint[0])
    plt.legend()
    #plt.show()

    Lhf_x = Lhf_x_func(eps_var=h_eps)

    print('Lf_x = ' + str(Lf_x))
    print('Lh_x = ' + str(Lh_x))
    print('Lhf_x = ' + str(Lhf_x))
    print('Ld = ' + str(Ld))
    print('dt = '+str(dt))

    print('c1 = ' + str(true_model.c1))
    print('c2 = ' + str(true_model.c2))
    print('h_eps  =' + str(h_eps))
    print('xr = ' + str(xr_setpoint))

    #plt.show()

    # Define robustness margin terms for the safety filter
    params_nonrob = params.copy()  # Used for non robust safety filter
    params['robust_margins'] = {'Lh_x': Lh_x, 'Ld': Ld, 'Lf_x': Lf_x,
                                'Lhf_x': Lhf_x}  # Robustness margin for handling perturbations



    # -----------------------------
    # Implementation
    options = {}
    options['use_feas'] = False
    options['time-varying'] = False
    options['event-trigger'] = True
    options_nonrob = {}
    options_nonrob['time-varying'] = False
    options_nonrob['event-trigger'] = False

    # Set up the safety filter and nominal control
    x0 = np.array(model.x0).reshape((nx, 1))  # 4.0 + model.x0.reshape((nx, 1)) # 23.5*np.ones((nx,1)) #
    SF = sf.SafetyFilter(x0=x0, f=model, params=params, constraints=constraints, options=options)
    SF_nonrob = sf.SafetyFilter(x0=x0, f=model, params=params_nonrob, constraints=constraints, options=options_nonrob)
    dummy_control = DummyControl()

    # Compute constraints with robust margins and plot to test feasibility
    X_rob_upper_term = []
    X_rob_lower_term = []
    for kk in range(Nsim):
        X_rob_upper_term.append(hf_level_plus(0) - SF.compute_robust_margin_terminal())
        X_rob_lower_term.append(hf_level_minus(0) + SF.compute_robust_margin_terminal())

    t = range(Nsim)
    plt.figure()
    plt.plot(t, model.xmax[-1]*np.ones(len(t)), 'k')
    plt.plot(t, model.xmin[-1]*np.ones(len(t)), 'k')
    plt.plot(X_rob_upper_term, 'b--')
    plt.plot(X_rob_lower_term, 'b')

    for ii in range(0, Nsim, 50):
        time_x = []
        X_rob_upper = []
        X_rob_lower = []
        for kk in range(params['N']):
            time_x.append(ii + kk)
            X_rob_upper.append(model.xmax[-1] - SF.compute_robust_margin(kk))
            X_rob_lower.append(model.xmin[-1] + SF.compute_robust_margin(kk))

        plt.plot(time_x, X_rob_upper, 'r--')
        plt.plot(time_x, X_rob_lower, 'r')

    plt.xlabel('Time step')
    plt.ylabel('State')
    plt.show()

    # Construct point-wise optimal control for staying in sublevel set of hf and check barrier condition
    ubar = ValidControl(c1=model.c1, c2=model.c2, dt=dt, r=xr_setpoint, rho=p2)
    x_range = np.linspace(xr_setpoint[-1] - sqrt(h_eps), xr_setpoint[-1] + sqrt(h_eps), 100)
    print('x_grid from x1,2 = '+str(xr_setpoint[-1] - sqrt(h_eps))+' to x1,2 = '+str(xr_setpoint[-1] + sqrt(h_eps)))
    check_barrier = TestBarrierFunction(model=model,u=ubar, dt=dt, rob_marg=SF.compute_robust_margin_terminal(), hf=hf)
    check_barrier.eval(x1_range=x_range, x2_range=x_range)



    CL = ClosedLoopSystem(f=true_model, u=SF, u_nom=u_DPC, dt=dt, int_type=policy_params['integrator_type'])
    CLtest = ClosedLoopSystem(f=true_model, u=dummy_control, u_nom=u_DPC, dt=dt, int_type=policy_params['integrator_type'])
    CLnonrob = ClosedLoopSystem(f=true_model, u=SF_nonrob, u_nom=u_DPC, dt=dt, int_type=policy_params['integrator_type'])
    #CLtest = ClosedLoopSystem(f=true_model, u=dummy_control, u_nom=ubar, dt=dt, int_type=policy_params['integrator_type']) # Just used to test ubar is correct

    start_time = time.time()
    [X, U] = CL.simulate(x0=x0, N=Nsim)
    end_time = time.time()
    CLtime = end_time - start_time

    start_time = time.time()
    [Xnonrob, Unonrob] = CLnonrob.simulate(x0=x0, N=Nsim)
    end_time = time.time()
    CLnonrob_time = end_time - start_time

    Nsim = X.shape[0]  # Correct Nsim in case closed-loop simulator interrupted by infeasible control action
    [Xtest, Utest] = CLtest.simulate(x0=x0, N=Nsim)

    print('Closed-loop simulation time (proposed) = ' + str(CLtime) + ' s')
    print('Closed-loop simulation time (non-rob) = ' + str(CLnonrob_time) + ' s')

    t = [kk for kk in range(Nsim+1)]
    plt.figure()
    plt.plot(t, [model.xmax[-1] for kk in range(Nsim + 1)], 'k', linewidth=3.0)
    plt.plot(t, [model.xmin[-1] for kk in range(Nsim + 1)], 'k', linewidth=3.0)
    plt.plot(np_refs, color='navy', label='ref', linewidth=3.0)
    plt.plot(Xtest[:, 0], color='crimson', label='x1 (DPC)', linewidth=2.0)
    plt.plot(Xtest[:, 1],  color='chartreuse', label='x2 (DPC)', linewidth=2.0)
    plt.plot(X[:, 0], color = 'orange', linestyle='--', label='x1 (DPC+SF)', linewidth=2.0)
    plt.plot(X[:, 1], color='magenta', linestyle='--', label='x2 (DPC+SF)', linewidth=2.0)
    plt.xlabel('Time step')
    plt.ylabel('Tank Level')
    plt.legend(loc='lower left')



    plt.figure()
    for ii in range(nu):
        plt.step(range(U.shape[0]), umax[ii] * np.ones(U.shape[0]), 'k', linewidth=3.0)
        plt.step(range(U.shape[0]), umin[ii] * np.ones(U.shape[0]), 'k', linewidth=3.0)
    plt.plot(Utest[:,0], color='crimson', label="u1 (DPC)", linewidth=2.0)
    plt.plot(Utest[:, 1], color='chartreuse', label="u2 (DPC)", linewidth=2.0)
    plt.plot(U[:, 0], color='orange', linestyle='--', label="u1 (DPC + SF)", linewidth=2.0)
    plt.plot(U[:, 1], color='magenta', linestyle='--', label="u2 (DPC + SF)", linewidth=2.0)
    plt.xlabel('Time step')
    plt.ylabel('Pump and Valve (Inputs)')
    plt.legend(loc='upper right', bbox_to_anchor=(0.77, 0.95))

    plt.figure()
    plt.step(range(Nsim + 1), Utest, 'b', label="U (DPC)")
    # plt.step(range(Nsim + 1), Utest, 'g--', label="Usf test")
    for ii in range(nu):
        plt.step(range(Nsim + 1), umax[ii] * np.ones(Nsim + 1), 'k--')
        plt.step(range(Nsim + 1), umin[ii] * np.ones(Nsim + 1), 'k--')
    plt.xlabel('Time step')
    plt.ylabel('Pump and Valve (Inputs)')
    plt.legend(loc='lower right')

    # Plot events
    if options['event-trigger']:
        plt.figure(figsize=(6.4, 1))
        plt.plot(SF.events, label='(DPC+SF)', linewidth=2.0)
        plt.xlabel('Time step')
        plt.ylabel('Events')
        print(SF.events_log)
        plt.legend() #plt.legend(loc='upper right', bbox_to_anchor=(0.71, 0.1))

    # Plot slack terms
    plt.figure()
    plt.plot(SF.slacks, label='State slacks')
    plt.plot(SF.slacks_term, label='Terminal slack')
    plt.xlabel('Time (s)')
    plt.ylabel('Slack values')
    plt.legend()

    # Plot comparison between old sf and proposed
    plt.figure()
    ax = plt.gca()
    plt.plot(t, [model.xmax[-1] for kk in range(Nsim + 1)], 'k', linewidth=3.0)
    plt.plot(t, [model.xmin[-1] for kk in range(Nsim + 1)], 'k', linewidth=3.0)
    plt.plot(np_refs, color='navy', label='ref', linewidth=3.0)
    plt.plot(Xtest[:, 0], color='crimson', label='x1 (DPC)', linewidth=2.0)
    plt.plot(Xtest[:, 1], color='chartreuse', label='x2 (DPC)', linewidth=2.0)
    plt.plot(X[:, 0], color='orange', linestyle='--', label='x1 (DPC+SF)', linewidth=2.0)
    plt.plot(X[:, 1], color='magenta', linestyle='--', label='x2 (DPC+SF)', linewidth=2.0)
    plt.xlabel('Time step')
    plt.ylabel('Tank Level')
    plt.legend(loc='lower left')
    ax.set_xlim([460, 515])
    ax.set_ylim([0.196, 0.206])

    plt.figure()
    ax = plt.gca()
    plt.plot(t, [model.xmax[-1] for kk in range(Nsim + 1)], 'k', linewidth=3.0)
    plt.plot(t, [model.xmin[-1] for kk in range(Nsim + 1)], 'k', linewidth=3.0)
    plt.plot(np_refs, color='navy', label='ref', linewidth=3.0)
    plt.plot(Xtest[:, 0], color='crimson', label='x1 (DPC)', linewidth=2.0)
    plt.plot(Xtest[:, 1], color='chartreuse', label='x2 (DPC)', linewidth=2.0)
    plt.plot(Xnonrob[:, 0], color='orange', linestyle='--', label='x1 (DPC+SF)', linewidth=2.0)
    plt.plot(Xnonrob[:, 1], color='magenta', linestyle='--', label='x2 (DPC+SF)', linewidth=2.0)
    plt.xlabel('Time step')
    plt.ylabel('Tank Level')
    plt.legend(loc='lower left')
    ax.set_xlim([460, 515])
    ax.set_ylim([0.196, 0.206])

    # Plot phases
    plt.figure()
    ax = plt.gca()
    ax.add_patch(patch.Rectangle((model.xmin[0], model.xmin[0]), max(model.xmax)-min(model.xmin), max(model.xmax)-min(model.xmin) ,fill=False,edgecolor = 'black', lw=3.0) )
    plt.plot(X[:, 0], X[:, 1], color='orange', linewidth=2.0, label='DPC+SF')
    plt.plot(Xnonrob[:, 0], Xnonrob[:, 1], color='cyan', linestyle='--', linewidth=2.0, label='DPC+SF non-robust')
    plt.plot(Xtest[:, 0], Xtest[:, 1], color='crimson', linestyle=':', linewidth=2.0, label='DPC')
    # x_range = np.linspace(0,1,50)
    # hlevel = np.zeros((len(x_range), len(x_range)))
    # for ii, x1 in enumerate(xr_range):
    #     for jj, x2 in enumerate(x_range):
    #         hlevel[ii,jj] = hf(np.array([x1,x2]).reshape((nx,1)),0)
    # plt.contour(x_range, x_range, hlevel, [0], color='k', linewidth=3.0)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 1.1])
    plt.legend()

    plt.figure()
    ax = plt.gca()
    ax.add_patch(patch.Rectangle((model.xmin[0], model.xmin[0]), max(model.xmax) - min(model.xmin),
                                 max(model.xmax) - min(model.xmin), fill=False, edgecolor='black', lw=3.0))
    plt.plot(X[:, 0], X[:, 1], color='orange', linewidth=2.0, label='DPC+SF')
    plt.plot(Xnonrob[:, 0], Xnonrob[:, 1], color='cyan', linestyle='--', linewidth=2.0, label='DPC+SF non-robust')
    plt.plot(Xtest[:, 0], Xtest[:, 1], color='crimson', linestyle=':', linewidth=2.0, label='DPC')
    # x_range = np.linspace(0,1,50)
    # hlevel = np.zeros((len(x_range), len(x_range)))
    # for ii, x1 in enumerate(xr_range):
    #     for jj, x2 in enumerate(x_range):
    #         hlevel[ii,jj] = hf(np.array([x1,x2]).reshape((nx,1)),0)
    # plt.contour(x_range, x_range, hlevel, [0], color='k', linewidth=3.0)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    ax.set_xlim([0.194, 0.207])
    ax.set_ylim([0.35, 0.68])
    plt.legend()

    plt.show()