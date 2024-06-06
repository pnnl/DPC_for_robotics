from neuromancer.psl import systems

from predictive_safety_filter import safety_filter as sf
import matplotlib.pyplot as plt
from closed_loop import ClosedLoopSystem
from closed_loop import DummyControl
from closed_loop import DPC_control
import math
from casadi import *
import torch



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

    model_name = 'SimpleSingleZone' # 'Reno_full' # 'Old_full' #'RenoLight_full' #
    model_type = 'Linear'

    # Define dynamics and parameters
    model = systems[model_type+model_name]() # RENOFULL, OR any RENO

    A = model.A
    B = model.Beta
    E = model.E
    G = model.G
    C = model.C
    F = model.F
    y_ss = model.y_ss
    nx = model.nx
    ny = model.ny
    nu = model.nq
    nd = E.shape[1]

    print('yss = '+str(model.y_ss))

    print('nx = '+str(nx))
    print('nu = '+str(nu))
    print('ny = '+str(ny))
    print('nd = '+str(nd))
    print('F = '+str(F))
    np.set_printoptions(suppress=True)
    print('A = '+str(A))
    print('B = '+str(B))
    print('C = '+str(C))
    np.set_printoptions(suppress=False)



    out = model.simulate()
    U = out['U'] if 'U' in out.keys() else None
    D = out['Dhidden'] if 'D' in out.keys() else None
    Dobs = out['D'] if 'D' in out.keys() else None

    dt = model.ts # sampling time
    #f = lambda x, t, u, A=A, B=B, G=G: A@x + B@u + G  # dx/dt = f(x,u)

    d = lambda x, t, u, E=E, D=D, nd=nd: E@np.array(D[t,:]).reshape((nd,1))  # known perturbation
    #f_pert = lambda x, t, u, f=f, d=d: f(x, t, u) + d(x, t, u)  # perturbed dynamics
    f = lambda x, t, u, A=A, B=B, G=G, E=E, D=D, nd=nd: A@x + B@u + G + E@np.array(D[t,:]).reshape((nd,1))  # dx/dt = f(x,u)
    dhidden_max = 0.005 #0.0 #
    dhidden = lambda x, t, u: -dhidden_max*np.ones((nx,1)) #unknown perturbation
    f_pert = lambda x, t, u, f=f: f(x, t, u) + dhidden(x, t , u)  # perturbed dynamics
    N = 20 #10  #   number of control intervals
    Nsim = 1000 #200 # Simulation time steps

    building_disturbance = np.array([E@np.array(D[ti,:]).reshape((nd,1)) for ti in range(Nsim)]).reshape(Nsim, -1)
    plt.figure()
    plt.plot(building_disturbance, linewidth=2.0)
    plt.xlabel('Time step')
    plt.ylabel(r'Disturbance Estimate, $\hat{w}(k)$')
    plt.show()

    # -----------------------------
    # Define parameters for the safety filter
    params = {}
    params['N'] = N  # prediction horizon
    params['dt'] = dt  # sampling time
    params['delta'] = 0.0 # Small robustness margin for constraint tightening
    params['alpha2='] = 1000000.0  # Multiplication factor for feasibility parameters
    params['alpha2'] = 1.0  # Multiplication factor for feasibility parameters
    params['integrator_type'] = 'cont'  # Integrator type for the safety filter

    # -----------------------------
    # Load the DPC policy for implementation
    policy_name = 'SimpleSingleZone_Linear_5'
    policy = torch.load(policy_name + '_policy.pth')
    policy_params = torch.load(policy_name + '_params.pth')

    # -----------------------------
    # Define system constraints
    terminal_constraint = {}
    state_constraints = {}
    input_constraints = {}
    params['nx'] = nx
    params['nu'] = nu
    params['ny'] = ny
    umax = np.array([model.umax]).flatten() #*np.ones((nu,1)) #[model.umax]  # Input constraint parameters(max)
    umin = np.array([model.umin]).flatten() #model.umin*np.ones((nu,1)) #[model.umin] # Input constraint parameters (min)

    amp = 1
    period = 300
    scu_k = SquareWave(period=period,amp=amp,offset=policy_params['Cup']+amp)
    scl_k = SquareWave(period=period,amp=amp,offset=policy_params['Clow']-amp,t_offset=pi)
    upper = []
    lower = []
    time = []
    for kk in range(Nsim):
        upper.append(scu_k.eval(kk))
        lower.append(scl_k.eval(kk))
        time.append(kk)


    # State constraint functions: g(x) <= 0
    kk = 0
    for ii in range(ny):
        ei = DM.zeros(ny,1)
        ei[ii] = 1.0
        state_constraints[kk] = lambda x, k, ei=ei, C=C, F=F, y_ss=y_ss, scu_k=scu_k, ny=ny: (ei.T @ (C @ x + F - y_ss * np.ones((ny, 1)))) - scu_k.eval(k)
        kk += 1
        state_constraints[kk] = lambda x, k, ei=ei, C=C, F=F, y_ss=y_ss, scl_k=scl_k, ny=ny: scl_k.eval(k) - (ei.T @ (C @ x + F - y_ss * np.ones((ny, 1))))
        kk += 1



    # Terminal constraint set
    yr_setpoint = 0.5*(policy_params['Cup'] + policy_params['Clow'])
    yr = yr_setpoint*np.ones((ny,1)) #[23.5]
    h_eps = 0.9 #0.1
    Cbar = np.vstack( (np.zeros((nx-ny, nx)), C) ) #np.vstack( (np.eye(nx-ny, nx), C) ) #
    abar = np.vstack( (np.zeros((nx-ny,1)), F - (y_ss+yr)*np.ones((ny,1)) ) )
    print('Cbar = '+str(Cbar))
    print('abar = '+str(abar))
    #hf = lambda x, C=C, F=F, y_ss=y_ss, yr=yr, ny=ny, h_eps=h_eps: (C@x + F - y_ss*np.ones((ny,1)) - yr).T @ (C@x + F - y_ss*np.ones((ny,1)) - yr) - h_eps
    hf = lambda x,k, Cbar=Cbar, abar=abar, h_eps=h_eps: (Cbar @ x + abar).T @ (Cbar @ x + abar) - h_eps
    hf_y = lambda y, y_ss=y_ss, yr=yr, h_eps=h_eps: (y - y_ss - yr_setpoint)**2 - h_eps

    print('F = '+str(F))
    print('y_ss = '+str(y_ss))
    print('yr ='+str(yr))
    #raise(SystemExit)

    # Find the level sets of the terminal constraints
    hf_level_plus = lambda a:  yr_setpoint + sqrt(h_eps + a)
    hf_level_minus = lambda a:  yr_setpoint - sqrt(h_eps + a)

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
    Lf_x = np.linalg.norm(A, ord=np.inf)
    Lh_x = np.linalg.norm(C, ord=np.inf)
    Lhf_x = 60.0 + 2 * np.linalg.norm(abar)
    Ld = dhidden_max

    print('Lf_x = ' + str(Lf_x))
    print('Lh_x = ' + str(Lh_x))
    print('Lhf_x = '+ str(Lhf_x))
    print('Ld = '  + str(Ld))
    print('C  = ' + str(C))
    print('A = ' + str(A))
    print('G = ' + str(G))

    params['robust_margins'] = {'Lh_x': Lh_x, 'Ld': Ld, 'Lf_x': Lf_x,
                                'Lhf_x': Lhf_x}  # Robustness margin for handling perturbations

    #-----------------------------
    # Construct the DPC policy
    # u_DPC = DPC_control(nu, policy, policy_params, {'D': D}, umin=model.umin, umax=model.umax )
    Ymax = torch.tensor(upper).type(torch.float)
    Ymin = torch.tensor(lower).type(torch.float)
    u_DPC = DPC_control(nu, policy, policy_params, {'ymin': Ymax[None,:], 'ymax': Ymin[None,:], 'D': torch.from_numpy(D.T).type(torch.float)}, umin=model.umin,
                        umax=model.umax, norm_mean=policy_params['means'], norm_stds=policy_params['stds'])

    # -----------------------------
    # Implementation
    options = {}
    options['use_feas'] = False

    # Set up the safety filter and nominal control
    x0 = 0.9*model.x0.reshape((nx, 1)) #4.0 + model.x0.reshape((nx, 1)) # 23.5*np.ones((nx,1)) #
    SF = sf.SafetyFilter(x0=x0, f=f, params=params, constraints=constraints, options=options)
    #u_nom = lambda x, k, nu=nu: np.zeros((nu,1)) #max(umax)*np.ones((nu,1))#np.array([[0]])#np.expand_dims(np.array([100.0*sin(k)]), 0)
    dummy_control = DummyControl()



    # Compute constraints with robust margins and plot to test feasibility
    X_rob_upper_term = []
    X_rob_lower_term = []
    X_upper_term = []
    X_lower_term = []
    for kk in range(Nsim):
        X_rob_upper_term.append(hf_level_plus(0) - SF.compute_robust_margin_terminal())
        X_rob_lower_term.append(hf_level_minus(0) + SF.compute_robust_margin_terminal())
        X_upper_term.append(hf_level_plus(0) )
        X_lower_term.append(hf_level_minus(0))

    plt.figure()
    plt.plot(time, upper, 'k', linewidth=3.0)
    plt.plot(time, lower, 'k', linewidth=3.0)
    plt.plot(X_upper_term, color='blue', linewidth=2.0)
    plt.plot(X_lower_term, color='blue', linewidth=2.0)
    plt.plot(X_rob_upper_term, 'c:', linewidth=2.0)
    plt.plot(X_rob_lower_term, 'c:', linewidth=2.0)

    for ii in range(0, Nsim, 50):
        time_x = []
        X_rob_upper = []
        X_rob_lower = []
        for kk in range(params['N']):
            time_x.append(ii+kk)
            X_rob_upper.append(scu_k.eval(ii+kk) - SF.compute_robust_margin(kk))
            X_rob_lower.append(scl_k.eval(ii+kk) + SF.compute_robust_margin(kk))

        plt.plot(time_x, X_rob_upper, color='#A0A0A0', linestyle='--', linewidth=2.0)
        plt.plot(time_x, X_rob_lower, color='#A0A0A0', linestyle='--', linewidth=2.0)

    plt.xlabel('Time step')
    plt.ylabel('Temperature (C)')
    plt.show()
    #raise(SystemExit)



    CL = ClosedLoopSystem(f=f_pert, u=SF, u_nom=u_DPC, dt=dt, int_type='cont')
    CLtest = ClosedLoopSystem(f=f_pert, u=dummy_control, u_nom=u_DPC, dt=dt, int_type='cont')
    [X, U] = CL.simulate(x0=x0, N=Nsim)

    Nsim = X.shape[0] -1 # Correct Nsim in case closed-loop simulator interrupted by infeasible control action
    [Xtest, Utest] = CLtest.simulate(x0=x0, N=Nsim)

    CLunpert = ClosedLoopSystem(f=f, u=SF, u_nom=u_DPC, dt=dt, int_type='cont')
    CLunperttest = ClosedLoopSystem(f=f, u=dummy_control, u_nom=u_DPC, dt=dt, int_type='cont')
    [Xunpert, Uunpert] = CLunpert.simulate(x0=x0, N=Nsim)
    [Xunperttest, Uunperttest] = CLunperttest.simulate(x0=x0, N=Nsim)

    Y = np.zeros((Nsim, ny))
    Ytest = np.zeros((Nsim, ny))
    Yunpert = np.zeros((Nsim, ny))
    Yunperttest = np.zeros((Nsim, ny))
    D_all = np.zeros((Nsim, nx))
    for ii in range(Nsim):
        Y[ii,:] = X[ii,:]@C.T + F.T - y_ss*np.ones((1,ny))
        D_all[ii,:] = d(X[ii,:], ii, U[ii,:]).flatten()
        Ytest[ii,:] = Xtest[ii,:]@C.T + F.T - y_ss*np.ones((1,ny))

        Yunpert[ii, :] = Xunpert[ii, :] @ C.T + F.T - y_ss * np.ones((1, ny))
        Yunperttest[ii, :] = Xunperttest[ii, :] @ C.T + F.T - y_ss * np.ones((1, ny))

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
    #plt.step(range(Y.shape[0]), hf_level_plus(0) * np.ones(Y.shape[0]), 'r--')
    #plt.step(range(Y.shape[0]), hf_level_minus(0) * np.ones(Y.shape[0]), 'r--')
    plt.plot(Y, 'g', label='Y (DPC+SF')
    plt.xlabel('Time step')
    plt.ylabel('Temperature (C)')
    plt.legend(loc='lower right')

    plt.figure()
    plt.plot(time, upper, 'k--')
    plt.plot(time, lower, 'k--')
    #plt.step(range(Y.shape[0]), hf_level_plus(0) * np.ones(Y.shape[0]), 'r--')
    #plt.step(range(Y.shape[0]), hf_level_minus(0) * np.ones(Y.shape[0]), 'r--')
    plt.plot(Ytest, 'g', label='Y (DPC)')
    plt.xlabel('Time step')
    plt.ylabel('Temperature (C)')
    plt.legend(loc='lower right')


    plt.figure()
    plt.step(range(U.shape[0]), U, 'b', label="U (DPC+SF)")
    #plt.step(range(Nsim + 1), Utest, 'g--', label="Usf test")
    for ii in range(nu):
        plt.step(range(U.shape[0]), umax[ii] * np.ones(U.shape[0]), 'k--')
        plt.step(range(U.shape[0]), umin[ii] * np.ones(U.shape[0]), 'k--')
    plt.xlabel('Time step')
    plt.ylabel('Heat flow (W)')
    plt.legend(loc='lower right')

    plt.figure()
    plt.step(range(U.shape[0]), Utest, 'b', label="U (DPC)")
    # plt.step(range(Nsim + 1), Utest, 'g--', label="Usf test")
    for ii in range(nu):
        plt.step(range(U.shape[0]), umax[ii] * np.ones(U.shape[0]), 'k--')
        plt.step(range(U.shape[0]), umin[ii] * np.ones(U.shape[0]), 'k--')
    plt.xlabel('Time step')
    plt.ylabel('Heat flow (W)')
    plt.legend(loc='lower right')

    plt.figure()
    plt.plot( D_all, label="D")
    plt.xlabel('Time step')
    plt.ylabel('Perturbations')
    plt.legend(loc='lower right')


    # Plot events
    if options['event-trigger']:
        plt.figure()
        plt.plot(SF.events)
        plt.xlabel('Time step')
        plt.ylabel('Events')
        print(SF.events_log)

    # Plot slack terms
    plt.figure()
    plt.plot(SF.slacks, label='State slacks')
    plt.plot(SF.slacks_term, label='Terminal slack')
    plt.xlabel('Time (s)')
    plt.ylabel('Slack values')
    plt.legend()

    # Combined plots
    plt.figure()
    plt.plot(time, upper, 'k', linewidth=3.0)
    plt.plot(time, lower, 'k', linewidth=3.0)
    plt.plot(Y, color='crimson', label='(DPC+SF perturbed)', linewidth=2.0)
    plt.plot(Ytest, color='orange', linestyle='--', label='(DPC perturbed)', linewidth=2.0)
    plt.plot(Yunpert, color='chartreuse', label='(DPC+SF unperturbed)', linewidth=2.0)
    plt.plot(Yunperttest, 'm--', label='(DPC unperturbed)', linewidth=2.0)
    plt.xlabel('Time step')
    plt.ylabel('Temperature (C)')
    plt.legend()

    plt.figure()
    plt.plot(time, lower, 'k', linewidth=3.0)
    plt.plot(Y, color='crimson', label='(DPC+SF perturbed)', linewidth=2.0)
    plt.plot(Ytest, color='orange', linestyle='--', label='(DPC perturbed)', linewidth=2.0)
    plt.plot(Yunpert, color='chartreuse', label='(DPC+SF unperturbed)', linewidth=2.0)
    plt.plot(Yunperttest, 'm--', label='(DPC unperturbed)', linewidth=2.0)
    plt.xlabel('Time step')
    plt.ylabel('Temperature (C)')
    axes=plt.gca()
    axes.set_ylim([17.25, 19])
    plt.legend(loc='upper right')

    plt.figure()
    plt.plot(U, color='crimson', label='(DPC+SF perturbed)', linewidth=2.0)
    plt.plot(Utest, color='orange', linestyle='--', label='(DPC perturbed)', linewidth=2.0)
    plt.plot(Uunpert, color='chartreuse', label='(DPC+SF unperturbed)', linewidth=2.0)
    plt.plot(Uunperttest, 'm--', label='(DPC unperturbed)', linewidth=2.0)
    for ii in range(nu):
        plt.step(range(U.shape[0]), umax[ii] * np.ones(U.shape[0]), 'k', linewidth=3.0)
        plt.step(range(U.shape[0]), umin[ii] * np.ones(U.shape[0]), 'k', linewidth=3.0)
    plt.xlabel('Time step')
    plt.ylabel('Heat flow (W)')
    plt.legend(loc='best', bbox_to_anchor=(0.55, 0.65, 0.5, 0.5))

    plt.show()


