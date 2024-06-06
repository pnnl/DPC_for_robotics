


import numpy as np
from predictive_safety_filter import safety_filter as sf
import matplotlib.pyplot as plt
from closed_loop import DummyControl
from closed_loop import ClosedLoopSystem


class Constraints():
    def __init__(self, xmax, xmin, umax, umin):
        self.xmax = xmax
        self.xmin = xmin
        self.umax = umax
        self.umin = umin


if __name__ == "__main__":


    # # #  VTOL aircraft model
    # System parameters
    m = 4  # mass of aircraft
    J = 0.0475  # inertia around pitch axis
    r = 0.25  # distance to center of force
    g = 9.8  # gravitational constant
    c = 0.05  # damping factor (estimated)
    # State space dynamics
    xe = [0, 0, 0, 0, 0, 0]  # equilibrium point of interest
    ue = [0, m * g]
    # model matrices
    A = np.array(
        [[0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 0, (-ue[0] * np.sin(xe[2]) - ue[1] * np.cos(xe[2])) / m, -c / m, 0, 0],
         [0, 0, (ue[0] * np.cos(xe[2]) - ue[1] * np.sin(xe[2])) / m, 0, -c / m, 0],
         [0, 0, 0, 0, 0, 0]]
    )
    # Input matrix
    B = np.array(
        [[0, 0], [0, 0], [0, 0],
         [np.cos(xe[2]) / m, -np.sin(xe[2]) / m],
         [np.sin(xe[2]) / m, np.cos(xe[2]) / m],
         [r / J, 0]]
    )
    # Output matrix
    C = np.array([[1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.]])
    D = np.array([[0, 0], [0, 0]])
    # reference
    x_ref = np.array([[0], [0], [0], [0], [0], [0]])
    # control equilibria
    u_ss = np.array([[0], [m * g]])
    # problem dimensions
    nx = A.shape[0]
    ny = C.shape[0]
    nu = B.shape[1]

    # -----------------------------
    # Define dynamics and parameters
    f = lambda x, t, u, A=A, B=B: A@x + B@u # dx/dt = f(x,u)
    N = 15 #  number of control intervals
    dt = 0.1  # sampling time

    # -----------------------------
    #  Define parameters for the safety filter
    params = {}
    params['N'] = N
    params['dt'] = dt
    params['delta'] = 0.00005  # Small robustness margin for constraint tightening
    params['robust_margin'] = 0.01  # 0.0#  Robustness margin for handling perturbations
    params['robust_margin_terminal'] = 0.0001  # Robustness margin for handling perturbations (terminal set)
    params['alpha2'] = 1.0  # Multiplication factor for feasibility parameters
    params['integrator_type'] = 'Euler'  # Integrator type for the safety filter
    Nsim = 120

    # -----------------------------
    # Define system constraints
    terminal_constraint = {}
    state_constraints = {}
    input_constraints = {}

    params['nx'] = nx
    params['nu'] = nu
    xmin = np.array([-0.8, -0.8, -0.8, -0.8, -0.8, -0.8]).reshape((nx,1)) #[-5, -5, -5, -5, -5, -5]
    xmax = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8]).reshape((nx,1))
    umin = np.array([-1, -1]).reshape((nu,1))
    umax = np.array([1, 1]).reshape((nu,1))


    kk = 0
    for ii in range(nx):
        state_constraints[kk] = lambda x, k, ii=ii, xmax=xmax: x[ii] - xmax[ii]
        kk += 1
        state_constraints[kk] = lambda x, k, ii=ii, xmin=xmin: xmin[ii] - x[ii]
        kk += 1
    P = np.eye(nx)
    xr = np.array(xe)
    hf = lambda x, P=P, xr=xr: (x - xr).T @ P @ (x - xr) - 0.01

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
    options['time-varying'] = False

    x0 = np.array([0.6, -.3, 0.1, 0.2, 0.4, 0.2]).reshape((nx,1))
    SF = sf.SafetyFilter(x0=x0, f=f, params=params, constraints=constraints, options=options)
    u_nom = lambda x, k, nu=nu: np.zeros((nu,1))  # 100.0*sin(0.2*k)
    dummy_control = DummyControl()

    if 1:
        # Run the closed-loop system, one with the safety-filter and the other only with the nominal control
        CL = ClosedLoopSystem(f=f, u=SF, u_nom=u_nom, dt=dt, int_type=params['integrator_type'])
        CLtest = ClosedLoopSystem(f=f, u=dummy_control, u_nom=u_nom, dt=dt, int_type=params['integrator_type'])
        [X, U] = CL.simulate(x0=x0, N=Nsim)
        [Xtest, Utest] = CLtest.simulate(x0=x0, N=Nsim)

        plt.figure()
        plt.plot(X[:, :], label='Xsf')
        #plt.plot(Xtest[:, :], 'r--', label='Xsf_test')

        plt.legend(loc='lower right')

        plt.figure()
        plt.step(range(Nsim + 1), U, 'b', label="Usf")
        #plt.step(range(Nsim + 1), Utest, 'g--', label="Usf test")
        for ii in range(nu):
            plt.step(range(U.shape[0]), umax[ii] * np.ones(U.shape[0]), 'k--')
            plt.step(range(U.shape[0]), umin[ii] * np.ones(U.shape[0]), 'k--')

        plt.legend(loc='lower right')

        plt.show()
    else:
        Xsf, Usf, Xi, XN = SF.open_loop_rollout(unom=u_nom, x0 = x0)
        # unom_f = lambda x, k: 0 #sin(k)
        # Xcl, Ucl = SF.closed_loop_simulation(unom_f=unom_f)

        plt.figure()
        for ii in range(nx):
            plt.plot(Xsf[ii, :], label='X'+str(ii))
        plt.legend(loc="upper right")
        plt.xlabel('Time (k)')
        plt.ylabel('x')
        plt.ylim([-0.8, 0.8])

        plt.figure()
        for ii in range(nu):
            plt.step(range(N), Usf[ii,:], label="U"+str(ii))
        plt.legend(loc="lower right")
        plt.xlabel('Time (k)')
        plt.ylabel('u')
        plt.ylim([-1.1, 1.1])

        plt.figure()
        plt.plot([np.linalg.norm(Xi[:, kk]) for kk in range(N-1)])
        plt.plot(N, XN, '*')
        plt.xlabel('Time (k)')
        plt.ylabel('Slack variables')
        plt.ylim([-0.001, 0.07])

        plt.show()