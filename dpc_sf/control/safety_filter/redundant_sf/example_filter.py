"""
Example of building from Wences
"""

from neuromancer.psl import systems, plot
from casadi import *
import safetyfilter as sf
import matplotlib.pyplot as plt
from safetyfilter import ClosedLoopSystem
from safetyfilter import DummyControl
import math
from casadi import *

class SquareWave():
    def __init__(self, period=1, amp=1, offset=0, t_offset=0):
        self.period = period
        self.amp = amp
        self.offset = offset
        self.t_offset = t_offset
    def eval(self, k):
        return self.amp*sign(sin(2.0*math.pi/self.period*k + self.t_offset)) + self.offset

if __name__ == '__main__':
    # -----------------------------

    model_name = 'SimpleSingleZone' #'Reno_full' #
    model_type = 'Linear'

    # Define dynamics and parameters
    model = systems[model_type+model_name]() # RENOFULL, OR any RENO

    A = model.A
    B = model.Beta
    E = model.E
    G = model.G
    C = model.C
    F = model.F
    nx = model.nx
    ny = model.ny
    nu = model.nq
    nd = E.shape[1]


    out = model.simulate()
    U = out['U'] if 'U' in out.keys() else None
    D = out['Dhidden'] if 'D' in out.keys() else None
    Dobs = out['D'] if 'D' in out.keys() else None

    dt = model.ts # sampling time
    f = lambda x, t, u: A@x + B@u + G  # dx/dt = f(x,u)
    d = lambda x, t, u: E@np.array(D[t]).reshape((nd,1))  # perturbation
    f_pert = lambda x, t, u: f(x, t, u) + d(x, t, u)  # perturbed dynamics
    N = 3  # number of control intervals
    Nsim = 1000 # Simulation time steps

    # -----------------------------
    # Define parameters for the safety filter
    params = {}
    params['N'] = N  # prediction horizon
    params['dt'] = dt  # sampling time
    params['delta'] = 0.00005  # Small robustness margin for constraint tightening
    params['robust_margin'] = 0.1 #0.0 #  #   Robustness margin for handling perturbations
    params['robust_margin_terminal'] = 0.0001 #0.0 # # Robustness margin for handling perturbations (terminal set)
    params['alpha2'] = 1.0  # Multiplication factor for feasibility parameters
    params['integrator_type'] = 'cont'  # Integrator type for the safety filter

    # -----------------------------
    # Define system constraints
    terminal_constraint = {}
    state_constraints = {}
    input_constraints = {}
    params['nx'] = nx
    params['nu'] = nu
    umax = [model.umax]  #model.umax*np.ones((nu,1)) # Input constraint parameters(max)
    umin = [model.umin] #model.umin*np.ones((nu,1)) # Input constraint parameters (min)


    scu_k = SquareWave(period=300,amp=2.5,offset=27.5)
    scl_k = SquareWave(period=300,amp=-2.5,offset=20)
    upper = []
    lower = []
    time = []
    for kk in range(Nsim):
        upper.append(scu_k.eval(kk))
        lower.append(scl_k.eval(kk))
        time.append(kk)
    # plt.figure()
    # plt.plot(time, upper)
    # plt.plot(time, lower)
    # plt.show()

    # State constraint functions: g(x) <= 0
    state_constraints[0] = lambda x, k: C@x + F - scu_k.eval(k)
    state_constraints[1] = lambda x, k: scl_k.eval(k) - C@x - F

    # Terminal constraint set
    yr_setpoint = 23.5
    yr = yr_setpoint*np.ones((ny,1)) #[23.5]
    hf = lambda x: (C@x + F - yr).T @ (C@x + F - yr) - 0.1

    # Input constraints: h(u) <= 0
    #input_constraints[0] = lambda u: u - umax
    #input_constraints[1] = lambda u: umin - u
    kk = 0
    for ii in range(nu):
        input_constraints[kk] = lambda u: u[ii] - umax[ii]
        kk += 1
        input_constraints[kk] = lambda u: umin[ii] - u[ii]
        kk += 1

    constraints = {}
    constraints['state_constraints'] = state_constraints
    constraints['hf'] = hf
    constraints['input_constraints'] = input_constraints

    # -----------------------------
    # Implementation
    options = {}
    # options['use_feas'] = False

    # Set up the safety filter and nominal control
    x0 = model.x0.reshape((nx, 1)) # 0.9*model.x0.reshape((nx, 1)) #
    SF = sf.SafetyFilter(x0=x0, f=f, params=params, constraints=constraints, options=options)
    u_nom = lambda x,k: np.zeros((nu,1)) #np.array([[0]])#np.expand_dims(np.array([100.0*sin(k)]), 0)
    dummy_control = DummyControl()

    CL = ClosedLoopSystem(f=f_pert, u=SF, u_nom=u_nom, dt=dt, int_type='cont')
    CLtest = ClosedLoopSystem(f=f_pert, u=dummy_control, u_nom=u_nom, dt=dt, int_type='cont')
    [X, U] = CL.simulate(x0=x0, N=Nsim)
    Nsim = X.shape[0] # Correct Nsim in case closed-loop simulator interrupted by infeasible control action
    [Xtest, Utest] = CLtest.simulate(x0=x0, N=Nsim)

    Y = np.zeros((Nsim, ny))
    D_all = np.zeros((Nsim, nx))
    for ii in range(Nsim):
        Y[ii,:] = X[ii,:]@C.T + F.T
        D_all[ii,:] = d(X[ii,:], ii, U[ii,:]).flatten()

    plt.figure()
    plt.plot(time, upper, 'k--')
    plt.plot(time, lower, 'k--')
    plt.plot(X[:, :], label='X')
    #plt.plot(Y, 'g')
    #plt.plot(Xtest[:, :], label='X_test')
    plt.xlabel('Time step')
    plt.ylabel('Temperature (C)')
    plt.legend(loc='lower right')

    plt.figure()
    plt.plot(time, upper, 'k--')
    plt.plot(time, lower, 'k--')
    plt.plot(Xtest[:, :], label='X_test')
    plt.xlabel('Time step')
    plt.ylabel('Temperature (C)')
    plt.legend(loc='lower right')

    plt.figure()
    plt.plot(time, upper, 'k--')
    plt.plot(time, lower, 'k--')
    plt.plot(Y, 'g', label='Y')
    plt.xlabel('Time step')
    plt.ylabel('Temperature (C)')
    plt.legend(loc='lower right')


    plt.figure()
    plt.step(range(U.shape[0]), U, 'b', label="U")
    #plt.step(range(Nsim + 1), Utest, 'g--', label="Usf test")
    plt.step(range(U.shape[0]), umax * np.ones(U.shape[0]), 'k--')
    plt.step(range(U.shape[0]), umin * np.ones(U.shape[0]), 'k--')
    plt.xlabel('Time step')
    plt.ylabel('Heat flow (W)')
    plt.legend(loc='lower right')

    plt.figure()
    plt.step(range(Nsim + 1), Utest, 'b', label="U_test")
    # plt.step(range(Nsim + 1), Utest, 'g--', label="Usf test")
    plt.step(range(Nsim + 1), umax * np.ones(Nsim + 1), 'k--')
    plt.step(range(Nsim + 1), umin * np.ones(Nsim + 1), 'k--')
    plt.xlabel('Time step')
    plt.ylabel('Heat flow (W)')
    plt.legend(loc='lower right')

    plt.figure()
    plt.plot( D_all, label="D")
    plt.xlabel('Time step')
    plt.ylabel('Perturbations')
    plt.legend(loc='lower right')

    plt.show()

    raise(SystemExit)


    out = model.simulate()
    Y = out['Y']
    X = out['X']
    U = out['U'] if 'U' in out.keys() else None
    print('U out= '+str(U))
    D = out['D'] if 'D' in out.keys() else None
    plot.pltOL(Y=Y, X=X, U=U, D=D)
    plot.pltPhase(X=Y)

