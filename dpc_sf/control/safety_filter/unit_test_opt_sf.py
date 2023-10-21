import numpy as np
import casadi as ca

from dpc_sf.dynamics.params import params as quad_params
from dpc_sf.dynamics.eom_ca import QuadcopterCA

from dpc_sf.control.safety_filter.safetyfilternew import SafetyFilter
from dpc_sf.control.safety_filter.safetyfilteroptimised import SafetyFilter as SafetyFilterOptimised

### basic parameters
nx = 17
nu = 4
N = 30     # prediction horizon No. steps
Nsim = 300  # simulation length

Ts = 0.01
Tf_hzn = 3.0
N = 300
Ti = 0.0
Tf = 15.0
integrator_type = "Euler"
obstacle_opts = {'r': 0.5, 'x': 1, 'y': 1}
# Find the optimal dts for the MPC
dt_1 = Ts
d = (2 * (Tf_hzn/N) - 2 * dt_1) / (N - 1)
dts = [dt_1 + i * d for i in range(N)]
### dummy policy
policy = lambda x, t: np.array([0.,0.,0.,0.])
### terminal constraint
xr = np.copy(quad_params["default_init_state_np"])
xr[0] = 2 # x  
xr[1] = 2 # y
xr[2] = 1 # z flipped
### create dynamics
quad = QuadcopterCA()
def f(x, t, u):
    return quad.state_dot_1d(x, u)[:,0]
### define SF parameters
params_opt = {}
params_opt['N'] = N  # prediction horizon
params_opt['Ts'] = Ts  # sampling time
params_opt['dts'] = dts
params_opt['delta'] = 0.00005  # Small robustness margin for constraint tightening
params_opt['robust_margin'] = 0.1 #0.0 #  #   Robustness margin for handling perturbations
params_opt['robust_margin_terminal'] = 0.0001 #0.0 # # Robustness margin for handling perturbations (terminal set)
params_opt['alpha2'] = 1.0  # Multiplication factor for feasibility parameters
params_opt['integrator_type'] = integrator_type  # Integrator type for the safety filter
params_opt['nx'] = nx
params_opt['nu'] = nu

params = {}
params['N'] = N  # prediction horizon
params['dt'] = Ts  # sampling time
params['delta'] = 0.00005  # Small robustness margin for constraint tightening
params['robust_margin'] = 0.1 #0.0 #  #   Robustness margin for handling perturbations
params['robust_margin_terminal'] = 0.0001 #0.0 # # Robustness margin for handling perturbations (terminal set)
params['alpha2'] = 1.0  # Multiplication factor for feasibility parameters
params['integrator_type'] = integrator_type  # Integrator type for the safety filt
params['nx'] = nx
params['nu'] = nu

### System constraints
terminal_constraint = {}
state_constraints = {}
input_constraints = {}
# state upper/lower bounds
umax = [quad.params["maxCmd"]]  #model.umax*np.ones((nu,1)) # Input constraint parameters(max)
umin = [quad.params["minCmd"]] #model.umin*np.ones((nu,1)) # Input constraint parameters (min)
scu_k = quad.params["state_ub"] 
scl_k = quad.params["state_lb"]
scu_k[0:13] *= 100
scl_k[0:13] *= 100
state_constraints[0] = lambda x, k: x - scu_k
state_constraints[1] = lambda x, k: scl_k - x
# cylinder constraint
x_c = 1.0
y_c = 1.0
r = 0.51
def cylinder_constraint(x, k):
    current_time = ca.sum1(dts[:k])
    multiplier = 1 + current_time * 0.1
    current_x, current_y = x[0], x[1]
    return r ** 2 * multiplier - (current_x - x_c)**2 - (current_y - y_c)**2
state_constraints[2] = cylinder_constraint# lambda x, k: r**2 * (1 + k * dt * 0.001) - (x[0] - x_c)**2 - (x[1] - y_c)**2
# Quadratic CBF (assuming linearisation valid)
cbf_const = (np.sqrt(2) - 0.5) ** 2
hf = lambda x: (x-xr).T @ (x-xr) - cbf_const
# hf = lambda x: r**2 - (x[0] - x_c)**2 - (x[1] - y_c)**2
# input constraints
kk = 0
nu = 4
for ii in range(nu):
    input_constraints[kk] = lambda u: u[ii] - umax[0]
    kk += 1
    input_constraints[kk] = lambda u: umin[0] - u[ii]
    kk += 1
# combine constraints to send to SF
constraints = {}
constraints['state_constraints'] = state_constraints
constraints['hf'] = hf
constraints['input_constraints'] = input_constraints
### instantiate safety filter
x0 = quad_params["default_init_state_np"]
SF_opt = SafetyFilterOptimised(x0=x0, f=f, params=params_opt, constraints=constraints, options={})
SF = SafetyFilter(x0=x0, f=f, params=params, constraints=constraints, options={})
# CL = sf.ClosedLoopSystem(f=f_pert, u=self.SF, u_nom=self.unom, dt=dt, int_type='Euler', use_true_unom=True)
u = SF_opt(x0, np.array([10.,0.,-10.,0.]))

unom = lambda x, k: np.array([10.,0.,-10.,0.])
u_test = SF.compute_control_step(unom, x0, 0)
print('fin')