
from casadi import *
from predictive_safety_filter import safety_filter as sf
import matplotlib.pyplot as plt


if __name__ == '__main__':

   #-----------------------------
   # Define dynamics and parameters
   f = lambda x, t, u: u - x  # dx/dt = f(x,u)
   N = 20 #50 # number of control intervals
   dt = 0.1 # sampling time

   # -----------------------------
   # Define parameters
   params = {}
   params['N'] = N
   params['dt'] = dt
   params['delta'] = 0.005

   # -----------------------------
   # Define system constraints
   terminal_constraint = {}
   state_constraints = {}
   input_constraints = {}
   nx = 1
   nu = 1
   params['nx'] = nx
   params['nu'] = nu
   xmin = [-1]
   xmax = [1.0]
   umin = [0]
   umax = [1]
   kk = 0
   for ii in range(nx):
      state_constraints[kk] = lambda x,k: x[ii] - xmax[ii]
      kk += 1
      state_constraints[kk] = lambda x,k: xmin[ii] - x[ii]
      kk += 1
   P = np.eye(nx)
   xr = [0.0]
   hf = lambda x: (x - xr).T@P@(x - xr) - 0.01

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
   #options['use_feas'] = False

   x0 = np.array([0.8]) # np.array([0, 0]) #
   SF = sf.SafetyFilter(x0=x0, f=f, params=params, constraints=constraints, options=options)
   Xsf, Usf, Xi, XN = SF.open_loop_rollout(unom = 0)
   #unom_f = lambda x, k: 0 #sin(k)
   #Xcl, Ucl = SF.closed_loop_simulation(unom_f=unom_f)

   plt.figure()
   plt.plot(Xsf, label='Xsf_pos')
   plt.step(range(N), Usf, 'b',label="Usf")
   plt.legend(loc="lower right")
   plt.xlabel('Time (k)')

   plt.figure()
   plt.plot([np.linalg.norm(Xi[:,kk]) for kk in range(N-1)])
   plt.plot(N, XN,'*')
   plt.xlabel('Time (k)')
   plt.ylim([0, max([0.05, XN,max([np.linalg.norm(Xi[:,kk]) for kk in range(N-1)]) ])])


   plt.show()
