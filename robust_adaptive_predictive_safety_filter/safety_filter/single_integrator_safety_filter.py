from casadi import *
from predictive_safety_filter import safety_filter as sf
import matplotlib.pyplot as plt
import torch
from closed_loop import DPC_control
from closed_loop import DummyControl
from closed_loop import ClosedLoopSystem
import time


def finding_bf( freq, dt, eps_var, Ld, rbar):

    xbar_func = lambda freq, dt, eps_var, Ld, rbar: rbar + sqrt(eps_var)
    Lh_x_func = lambda freq, dt, eps_var, Ld, rbar: 2.0*sqrt(eps_var)
    Lh_k_func = lambda freq, dt, eps_var, Ld, rbar: 2.0* xbar_func(freq, dt, eps_var, Ld, rbar) * rbar * freq + 2 * rbar ** 2 * freq
    umax_func = lambda freq, dt, eps_var, Ld, rbar: 1.0 / dt * (Lh_k_func(freq, dt, eps_var, Ld, rbar) + sqrt(eps_var)) - sqrt(eps_var - Lh_x_func(freq, dt, eps_var, Ld, rbar) * Ld) / dt
    Lf_func = lambda freq, dt, eps_var, Ld, rbar: dt * ( umax_func(freq, dt, eps_var, Ld, rbar))
    a_func = lambda freq, dt, eps_var, Ld, rbar: Lh_x_func(freq, dt, eps_var, Ld, rbar)*( Lf_func(freq, dt, eps_var, Ld, rbar) + Ld ) + Lh_k_func(freq, dt, eps_var, Ld, rbar)
    func = lambda freq, dt, eps_var, Ld, rbar: eps_var - a_func(freq, dt, eps_var, Ld, rbar)

    return func(freq, dt, eps_var, Ld, rbar)

if __name__ == "__main__":


    """
    # # #  Arguments, dimensions, bounds
    """
    Nsim = 250

    nx = 1
    nu = 1
    dt = 0.01 #0.001
    wbar = 0.02
    f = lambda x, t, u, dt=dt: u
    d = lambda x, t, u, dt=dt, wbar=wbar: wbar*sin(0.5*t)
    fpert  = lambda x, t, u, dt=dt, f=f, d=d: f(x,t,u) + d(x,t,u)

    #----------------------------
    # Reference trajectory
    rbar = 0.5
    freq = 0.05
    xr = lambda t, rbar=rbar: rbar*sin(freq*t)
    r_nsteps = Nsim
    R = torch.tensor([xr(k) for k in range(r_nsteps+1)], dtype=torch.float32).reshape(1, r_nsteps+1)

    # -----------------------------
    # Define parameters
    # -----------------------------
    # Define parameters for the safety filter
    params = {}
    params['N'] = 1  # prediction horizon
    params['dt'] = dt
    params['delta'] = 0.0 #0.000005  # Small robustness margin for constraint tightening
    params['alpha2='] = 1000000.0  # Multiplication factor for feasibility parameters
    params['alpha2'] = 1.0  # Multiplication factor for feasibility parameters
    params['integrator_type'] = 'Euler'  # Integrator type for the safety filter

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
    params['ny'] = nx

    eps_var = 0.2# 0.01
    xbar = rbar + sqrt(eps_var)
    Lh_x = 2.0 * sqrt(eps_var)
    Lh_k = 2*xbar*rbar*freq + 2*rbar**2*freq
    #umax = 1.0/dt*( xbar + dt*xbar + rbar) - sqrt(eps_var-Lh_x*wbar)/dt  #xbar + sqrt(eps_var)/dt #- sqrt(eps_var-Lh_x*wbar)/dt#ceil(xbar+ sqrt(eps_var)/dt)
    ubar = 1/dt*(Lh_k + sqrt(eps_var))- sqrt(eps_var-Lh_x*wbar)/dt
    umax = ceil(ubar)
    umin = -umax
    print('umax = '+str(umax))
    print('ubar = ' + str(ubar))



    # Terminal constraint set
    hf = lambda x, k, xr=xr, eps_var=eps_var: (x - xr(k))**2 - eps_var
    hf_level_plus = lambda a, t, eps_var=eps_var, xr=xr: xr(t) + sqrt(eps_var + a)
    hf_level_minus = lambda a, t, eps_var=eps_var, xr=xr: xr(t) - sqrt(eps_var + a)

    # Input constraints: h(u) <= 0
    input_constraints[0] = lambda u, umax=umax: u[0] - umax
    input_constraints[1] = lambda u, umin=umin: umin - u[0]

    constraints = {}
    constraints['state_constraints'] = state_constraints
    constraints['hf'] = hf
    constraints['input_constraints'] = input_constraints

    # Robustness margins

    Lf_x = 1.0
    Lf = dt*(umax)
    Ld = wbar
    a = Lh_x*(Lf + Ld) + Lh_k

    print('Lf_x = ' + str(Lf_x))
    print('Lf = ' + str(Lf))
    print('Lh_x = ' + str(Lh_x))
    print('Lh_k = ' + str(Lh_k))
    print('Ld = ' + str(Ld))
    print('a = ' + str(a))

    plt.figure()
    var = np.linspace(0, .5, 100)
    plt.plot(var, [finding_bf(freq=.05, dt=dt, eps_var=eps_var, Ld=wbar, rbar=vari) for vari in var])
    plt.show()


    params['robust_margins'] = {'Lh_x': Lh_x, 'Ld': Ld, 'Lf_x': Lf_x,
                                'Lhf_x': Lh_x, 'Lh_k':Lh_k, 'Lf':Lf, 'a': a}  # Robustness margin for handling perturbations

    # -------------------------------
    # Load DPC policy
    policy_name = 'single_integrator'
    version = 1
    policy = torch.load(policy_name + '_policy_' + str(version) + '.pth')
    policy_params = torch.load(policy_name + '_params_' + str(version) + '.pth')
    u_DPC = DPC_control(nu, policy, policy_params, {'r': R}, umin=umin, umax=umax)

    # -----------------------------
    # Implementation
    options = {}
    options['use_feas'] = False
    options['event-trigger'] = True

    # Set up the safety filter and nominal control
    x0 = [0.43] #[0]
    x0 = np.array(x0).reshape((nx, 1))
    SF = sf.OneStepSafetyFilter(x0=x0, f=f, params=params, constraints=constraints, options=options)
    dummy_control = DummyControl()
    u_nom = lambda x, k: 0
    #u_nom = lambda x, k, xr=xr, dt=dt, eps_var=eps_var, Lh_x=Lh_x, wbar=wbar, hf=hf, f=f: 0 if hf(f(x,k,0),k+1) <= -Lh_x*wbar else -x + 1.0 / dt * (-x + xr(k + 1)) - sign(-x + 1.0 / dt * (-x + xr(k + 1)))*sqrt(eps_var - Lh_x*wbar)/dt

    # Compute constraints with robust margins and plot to test feasibility
    X_rob_upper_term = []
    X_rob_lower_term = []
    for kk in range(Nsim):
        X_rob_upper_term.append(hf_level_plus(0,kk) - SF.compute_robust_margin_terminal())
        X_rob_lower_term.append(hf_level_minus(0,kk) + SF.compute_robust_margin_terminal())

    t = range(Nsim)
    plt.figure()
    plt.plot(t, [hf_level_plus(0,ti) for ti in t], 'b', linewidth=3.0)
    plt.plot(t, [hf_level_minus(0,ti) for ti in t], 'b', linewidth=3.0)
    plt.plot(t, [hf_level_plus(-a, ti) for ti in t], color='pink', linewidth=2.0)
    plt.plot(t, [hf_level_minus(-a, ti) for ti in t], color='pink', linewidth=2.0)
    plt.plot(X_rob_upper_term, 'c:', linewidth=2.0)
    plt.plot(X_rob_lower_term, 'c:', linewidth=2.0)

    print('terminal robust margin = ' + str(SF.compute_robust_margin_terminal()))
    plt.xlabel('Time step')
    plt.ylabel('State')
    plt.show()




    # Simulate the systems
    CL = ClosedLoopSystem(f=fpert, u=SF, u_nom=u_nom, dt=dt, int_type=params['integrator_type'])
    CLtest = ClosedLoopSystem(f=fpert, u=dummy_control, u_nom=u_nom, dt=dt, int_type=params['integrator_type'])
    Cldpc = ClosedLoopSystem(f=fpert, u=SF, u_nom=u_DPC, dt=dt, int_type=params['integrator_type'])

    start_time = time.time()
    [X, U] = CL.simulate(x0=x0, N=Nsim)
    end_time = time.time()
    CLtime = end_time - start_time
    CLevents = SF.events
    SF.clear_events()

    start_time = time.time()
    [Xtest, Utest] = CLtest.simulate(x0=x0, N=Nsim)
    end_time = time.time()
    CLtest_time = end_time - start_time
    SF.clear_events()

    start_time = time.time()
    [Xdpc, Udpc] = Cldpc.simulate(x0=x0,N=Nsim)
    end_time = time.time()
    CLdpc_time = end_time - start_time
    CLdpc_events = SF.events
    SF.clear_events()

    print('Closed-loop simulation time (CL) = ' + str(CLtime) + ' s')
    print('Closed-loop simulation time (CLtest) = ' + str(CLtest_time) + ' s')
    print('Closed-loop simulation time (CLdpc) = ' + str(CLdpc_time) + ' s')

    t = [kk for kk in range(Nsim+1)]
    plt.figure()
    plt.plot(t, [hf_level_plus(0,ti) for ti in t], 'b', linewidth=3.0)
    plt.plot(t, [hf_level_minus(0,ti) for ti in t], 'b', linewidth=3.0)
    plt.plot(t, [hf_level_plus(-a, ti) for ti in t], color='pink', linewidth=2.0)
    plt.plot(t, [hf_level_minus(-a, ti) for ti in t], color='pink', linewidth=2.0)
    plt.plot(X, color = 'crimson', label='(unom=0 + SF)', linewidth=2.0)
    plt.plot(Xtest[:], color = 'orange', linestyle='--', label='(unom=0)', linewidth=2.0)
    plt.plot(Xdpc[:], color='magenta', linestyle=':', label='(DPC + SF)', linewidth=2.0)
    plt.xlabel('Time step')
    plt.ylabel('State')
    axes = plt.gca()
    axes.set_ylim([-1, 1])
    plt.legend(loc='lower right')


    plt.figure()
    plt.step(range(U.shape[0]), U, color = 'crimson', label="(unom=0 + SF)", linewidth=2.0)
    plt.step(range(Nsim + 1), Utest,  color = 'orange', linestyle='--', label="(unom=0)", linewidth=2.0)
    plt.step(range(Nsim + 1), Udpc, color='magenta', linestyle=':', label="(DPC + SF)", linewidth=2.0)
    for ii in range(nu):
        plt.step(range(U.shape[0]), umax * np.ones(U.shape[0]), 'k', linewidth=3.0)
        plt.step(range(U.shape[0]), umin * np.ones(U.shape[0]), 'k', linewidth=3.0)
    plt.xlabel('Time step')
    plt.ylabel('Input')
    plt.legend(loc='lower right')

    # Plot slack terms
    plt.figure()
    plt.plot(SF.slacks_term, label='Terminal slack')
    plt.xlabel('Time (s)')
    plt.ylabel('Slack values')
    plt.legend()

    plt.figure()
    plt.plot(t, [d(0,ti,0) for ti in t])

    # Plot events
    if options['event-trigger']:
        plt.figure(figsize=(6.4, 1))
        plt.plot(CLevents, color = 'crimson', label="(unom=0 + SF)", linewidth=2.0)
        plt.plot(CLdpc_events, color='magenta', linestyle=':', label="(DPC + SF)", linewidth=2.0)
        plt.xlabel('Time step')
        plt.ylabel('Events')
        plt.legend()
        #print(SF.events_log)

    plt.show()