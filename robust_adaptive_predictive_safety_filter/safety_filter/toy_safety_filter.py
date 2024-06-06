
from casadi import *
from predictive_safety_filter import safety_filter as sf
import matplotlib.pyplot as plt
from predictive_safety_filter.safety_filter import ClosedLoopSystem
from predictive_safety_filter.safety_filter import DummyControl


if __name__ == '__main__':

    # Examples:
    #   Disturbance rejection:
    #       d = [0; 50sin(t)]
    #       robustness_margin = 0.1 (toggle btw 0.1 and 0 to see active and non-active disturbance rejection)


    #-----------------------------
    # Define dynamics and parameters
    dt = 0.01  # sampling time
    f = lambda x, t, u: vertcat(x[1] , - x[1] + u )  # dx/dt = f(x,u)
    d = lambda x, t, u: vertcat(0, 50.0*sin(t))     # perturbation
    f_pert = lambda x, t, u, f=f, d=d: f(x,t,u) + d(x,t,u)     # perturbed dynamics
    N = 40 # number of control intervals


    #-----------------------------
    # Define parameters for the safety filter
    params = {}
    params['N'] = N     # prediction horizon
    params['dt'] = dt   # sampling time
    params['delta'] = 0.00005   # Small robustness margin for constraint tightening
    params['robust_margin'] = 0.2  #0.0#  Robustness margin for handling perturbations
    params['robust_margin_terminal'] = 0.0001 # Robustness margin for handling perturbations (terminal set)
    params['alpha2'] = 1.0      # Multiplication factor for feasibility parameters
    params['integrator_type'] = 'Euler' # Integrator type for the safety filter

    #-----------------------------
    # Define system constraints
    terminal_constraint = {}
    state_constraints = {}
    input_constraints = {}
    nx = 2  # number of states
    nu = 1  # number of inputs
    params['nx'] = nx
    params['nu'] = nu
    xmax = [1.2, 3]  # State constraint parameters (max)
    xmin = [-1.2, -4] # State constraints parameters (min)
    umax = [30]     # Input constraint parameters(max)
    umin = [-30]    # Input constraint parameters (min)

    # State constraint functions
    upper_bound = lambda k, xmax=xmax: 0.8*sin(0.2*k) + xmax[0]
    lower_bound = lambda k, xmin=xmin: 0.8*sin(0.2*k) + xmin[0]
    state_constraints[0] = lambda x, k, upper_bound=upper_bound: x[0] - upper_bound(k)
    state_constraints[1] = lambda x, k, lower_bound=lower_bound: lower_bound(k) - x[0]
    state_constraints[2] = lambda x, k, xmax=xmax: x[1] - xmax[1]
    state_constraints[3] = lambda x, k, xmin=xmin: xmin[1] - x[1]

    # Terminal constraint set
    P = np.eye(nx)
    xr = [0.0, 0.0]
    hf = lambda x, P=P, xr=xr: (x - xr).T@P@(x - xr) - 0.001

    # Input constraints
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

    # -----------------------------
    # Implementation
    options = {}
    #options['use_feas'] = False

    # Set up the safety filter and nominal control
    x0 = np.array([1.0, 0])
    SF = sf.SafetyFilter(x0=x0, f=f, params=params, constraints=constraints, options=options)
    u_nom = lambda x,k: 0 #100.0*sin(0.2*k)
    dummy_control = DummyControl()

    if 1:
        # Run the closed-loop system, one with the safety-filter and the other only with the nominal control
        Nsim = 120
        CL = ClosedLoopSystem(f=f_pert, u=SF, u_nom=u_nom, dt=dt)
        CLtest = ClosedLoopSystem(f=f_pert, u=dummy_control, u_nom=u_nom, dt=dt)
        [X, U] = CL.simulate(x0 = x0, N=Nsim)
        [Xtest, Utest] = CLtest.simulate(x0 = x0, N=Nsim)

        plt.figure()
        plt.plot(X[:, 0], label='Xsf_pos')
        plt.plot(Xtest[:, 0], 'r--', label='Xsf_pos_test')
        plt.ylim([-3, 3])

        plt.plot([upper_bound(kk) for kk in range(Nsim)])
        plt.plot([lower_bound(kk) for kk in range(Nsim)])
        plt.legend(loc="lower right")

        plt.figure()
        plt.plot(X[:, 1], label='Xsf_speed')
        plt.plot(Xtest[:, 1], 'k--', label='Xsf_speed_test')
        plt.legend(loc='lower right')

        plt.figure()
        plt.step(range(Nsim+1), U, 'b', label="Usf")
        plt.step(range(Nsim + 1), Utest, 'g--', label="Usf test")
        plt.step(range(Nsim+1), umax * np.ones(Nsim+1), 'k')
        plt.step(range(Nsim+1), umin * np.ones(Nsim+1), 'k')
        plt.legend(loc='lower right')


        plt.show()


    else:
        # Run the open-loop rollout of the safety filter
        k0 = 0
        Xsf, Usf, Xi, XN = SF.open_loop_rollout(unom=u_nom, x0=x0, k0=k0)


        plt.figure()
        plt.plot([ kk for kk in range(k0, k0+N+1) ], Xsf[0,:], label='Xsf_pos')
        plt.ylim([-3, 3])

        plt.plot([ kk for kk in range(k0, k0+N) ], [ upper_bound(kk) for kk in range(k0, k0+N) ])
        plt.plot([ kk for kk in range(k0, k0+N) ], [lower_bound( kk) for kk in range(k0, k0+N)])
        plt.legend(loc="lower right")

        plt.figure()
        plt.plot([ kk for kk in range(k0, k0+N+1) ],Xsf[1, :], label='Xsf_speed')
        plt.legend(loc='lower right')

        plt.figure()
        plt.step(range(k0, k0+N), Usf, 'b', label="Usf")
        plt.step(range(k0, k0+N), umax*np.ones(N),'k' )
        plt.step(range(k0, k0+N), umin * np.ones(N),'k')
        plt.legend(loc='lower right')


        plt.figure()
        plt.plot([ kk for kk in range(k0, k0+N-1) ], [np.linalg.norm(Xi[:,kk]) for kk in range(N-1)])
        plt.plot(k0+N, XN,'*')
        plt.xlabel('Time (k)')
        plt.ylim([-.05, max([0.05, XN, max([np.linalg.norm(Xi[:, kk]) for kk in range(N - 1)])])])

        plt.show()
