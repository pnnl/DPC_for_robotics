import casadi as ca
from quad import Quadcopter
from utils import pytorch_utils as ptu
import numpy as np
import matplotlib.pyplot as plt

# step 0: get the quadcopter in here
quad = Quadcopter()
Ts = 0.1

# step 1: instantiate casadi optimisation
opti = ca.Opti()

# step 2: define variables to be optimised
N = 20

X = opti.variable(17, N+1)
U = opti.variable(4, N+1) # recall the final input plays no role, just makes concatenation in quad easier

# step 4: define system dynamics
test_sdot = quad.casadi_state_dot(X, U)
print(test_sdot)

# step 5: define start and end positions
state_start = quad.state.tolist()

state_ref = quad.state
state_ref[0] = 2 # same state of hover, 2 meters up in the z direction of where we started
XR = ca.MX(ptu.to_numpy(state_ref))
XR = ca.horzcat(*[XR]*(N+1))

# seems to be a much better way to do it
init = opti.parameter(17,1)
opti.set_value(init, state_start)
opti.subject_to(X[:,0] == init)

state_start_2 = state_start
state_start_2[0] = -2
opti.set_value(init, state_start_2)

# step 6: define dynamics
for k in range(N):
    sdot = quad.casadi_state_dot(X, U)
    opti.subject_to(X[:,k+1] == X[:,k] + sdot[:,k] * Ts)

# step 7: define input constraints
opti.subject_to(opti.bounded(-100, U, 100))

# step 8: define cost function
X_error = XR[:13,:] - X[:13,:]
cost = ca.sumsqr(U) + ca.sumsqr(X_error)
opti.minimize(cost)

# step 9: solve the NLP
opti.solver('ipopt')
sol = opti.solve()

# step 10: retrieve solution
u_opt = sol.value(U)[:,0]

print(sol.value(X))

print('fin')