from quad import sys_params

params, casadi_constraints, (casadi_state_lower_bound, casadi_state_upper_bound) = sys_params()

kTh = params["kTh"]
kTo = params["kTo"]
wmin = params["minWmotor"]
wmax = params["maxWmotor"]

min_thrust = kTh * wmin ** 2
min_torque = kTo * wmin ** 2
max_thrust = kTh * wmax ** 2
max_torque = kTo * wmax ** 2

# this is the last paramter of the motor gear parameter
tor_ratio = kTo/kTh

# we do everything relative to the thrust application: ctrlrange is  between minthrust and maxthrust
ctrlmin = min_thrust
ctrlmax = max_thrust

print(max_thrust)
