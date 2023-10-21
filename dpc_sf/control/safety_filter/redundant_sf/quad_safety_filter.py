from neuromancer.psl import systems, plot
from casadi import *
import safetyfilter as sf
import matplotlib.pyplot as plt
from safetyfilter import ClosedLoopSystem
from safetyfilter import DummyControl
import math
from dpc_sf.utils.random_state import random_state
from dpc_sf.utils.animation import Animator
from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.control.trajectory.trajectory import waypoint_reference
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
    from dpc_sf.dynamics.eom_ca import QuadcopterCA

    # instantiate the dynamics
    # ------------------------
    quad = QuadcopterCA()
    dt = 0.1
    nx = 17
    def f(x, t, u):
        return quad.state_dot_1d(x, u)[:,0]
    # f = lambda x, t, u: quad.state_dot_1d(x, u)[:,0]

    # disturbance on the input
    # ------------------------
    disturbance_scaling = 0.001
    # idx = [7,8,9,10,11,12] # select velocities to perturb
    idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] # select all states to perturb
    # idx = [14,15,16,17] # perturb just actuators
    # idx = [0,1,2] # perturb position
    # idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13] # perturb all but actuators
    pert_idx = [1 if i in idx else 0 for i in range(nx)]
    pert_mat = np.diag(pert_idx)
    d = lambda x, t, u: pert_mat * np.sin(t) * disturbance_scaling
    f_pert = lambda x, t, u: f(x, t, u) + d(x, t, u)
    N = 30
    Nsim = 50

    # Safety Filter Parameters
    # ------------------------
    params = {}
    params['N'] = N  # prediction horizon
    params['dt'] = dt  # sampling time
    #params['delta'] = 0.00005  # Small robustness margin for constraint tightening
    params['delta'] = 0.001
    params['robust_margin'] = 0.1 #0.0 #  #   Robustness margin for handling perturbations
    params['robust_margin_terminal'] = 0.0001 #0.0 # # Robustness margin for handling perturbations (terminal set)
    params['alpha2'] = 1.0  # Multiplication factor for feasibility parameters
    params['integrator_type'] = 'Euler'  # Integrator type for the safety filter

    # Define System Constraints
    # -------------------------
    terminal_constraint = {}
    state_constraints = {}
    input_constraints = {}
    params['nx'] = 17
    params['nu'] = 4
    umax = [quad.params["maxCmd"]]  #model.umax*np.ones((nu,1)) # Input constraint parameters(max)
    umin = [quad.params["minCmd"]] #model.umin*np.ones((nu,1)) # Input constraint parameters (min)

    # replacing the state upper and lower bounds
    #scu_k = SquareWave(period=1,amp=2.5,offset=27.5)
    #scl_k = SquareWave(period=1,amp=-2.5,offset=20)
    scu_k = quad.params["state_ub"]
    scl_k = quad.params["state_lb"]

    # upper = []
    # lower = []
    # time = []
    # for kk in range(Nsim):
    #     upper.append(scu_k.eval(kk))
    #     lower.append(scl_k.eval(kk))
    #     time.append(kk)

    # State constraint functions: g(x) <= 0
    # state_constraints[0] = lambda x, k: x - scu_k.eval(k)
    # state_constraints[1] = lambda x, k: scl_k.eval(k) - x
    state_constraints[0] = lambda x, k: x - scu_k
    state_constraints[1] = lambda x, k: scl_k - x
    # state_constraints[2] = lambda x, k: 

    # terminal constraint set
    xr = np.copy(quad_params["default_init_state_np"])
    xr[0] = 2
    xr[1] = 2
    xr[2] = 1

    # Quadratic CBF
    hf = lambda x: (x-xr).T @ (x-xr) - 0.1

    # Input constraints: h(u) <= 0
    #input_constraints[0] = lambda u: u - umax
    #input_constraints[1] = lambda u: umin - u
    kk = 0
    nu = 4
    for ii in range(nu):
        input_constraints[kk] = lambda u: u[ii] - umax[0]
        kk += 1
        input_constraints[kk] = lambda u: umin[0] - u[ii]
        kk += 1    

    constraints = {}
    constraints['state_constraints'] = state_constraints
    constraints['hf'] = hf
    constraints['input_constraints'] = input_constraints    

    # Implementation
    # --------------

    options = {}
    # options['use_feas'] = False

    # Set up the safety filter and nominal control
    # x0 = random_state()[:,None] # 0.9*model.x0.reshape((nx, 1)) #
    x0 = quad_params["default_init_state_np"]
    SF = sf.SafetyFilter(x0=x0, f=f, params=params, constraints=constraints, options=options)
    u_nom = lambda x,k: np.zeros((nu,1)) #np.array([[0]])#np.expand_dims(np.array([100.0*sin(k)]), 0)
    dummy_control = DummyControl()
    CL = ClosedLoopSystem(f=f_pert, u=SF, u_nom=u_nom, dt=dt, int_type='Euler')
    CLtest = ClosedLoopSystem(f=f_pert, u=dummy_control, u_nom=u_nom, dt=dt, int_type='Euler')
    [X, U] = CL.simulate(x0=x0, N=Nsim)

    # test open loop rollout
    Xopl, Uopl, Xi_sol, XiN_sol = SF.open_loop_rollout(unom=u_nom, x0=x0, k0=0)

    # lets get the animation in here
    t = np.linspace(0, Nsim*dt, Nsim)
    r = np.zeros(t.shape)
    R = waypoint_reference('wp_p2p', average_vel=1.0, include_actuators=True)

    render_interval = 1
    animator = Animator(
        states=X[::render_interval,:], 
        times=t[::render_interval], 
        reference_history=X[::render_interval], 
        reference=R, 
        reference_type='wp_p2p', 
        drawCylinder=True,
        state_prediction=None
    )
    animator.animate() # does not contain plt.show()    
    plt.show()

#     # Number of steps and states
#     num_steps = 51
#     num_states = 17
# 
#     # Create a subplot grid with the desired layout (e.g., 5 rows, 11 columns)
#     num_rows = 5
#     num_columns = 11
# 
#     # Loop through each step and plot it on the corresponding subplot
#     for i in range(50):
#         plt.subplot(num_rows, num_columns, i + 1)  # i+1 to start subplot index from 1
#         plt.plot(range(num_states), X[i, :], 'b.-')
#         plt.title('Step {}'.format(i + 1))
#         plt.xlabel('State')
#         plt.ylabel('Value')
#         plt.grid(True)
# 
#     plt.tight_layout()  # Adjust the layout for better spacing between subplots
#     plt.show()

    print('fin')


