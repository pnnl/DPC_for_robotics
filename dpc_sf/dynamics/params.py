import dpc_sf.utils.pytorch_utils as ptu
import dpc_sf.utils as utils
import numpy as np
import torch
import casadi as ca
from numpy import pi

# Quad Params
# ---------------------------
params = {}
params["mB"]   = 1.2       # mass (kg)

# modification to test differentiability improvement:
# params["mB"] = 0.5

params["g"]    = 9.81      # gravity (m/s/s)
params["dxm"]  = 0.16      # arm length (m)
params["dym"]  = 0.16      # arm length (m)
params["dzm"]  = 0.05      # motor height (m)
params["IB"]   = ptu.from_numpy(np.array(
                    [[0.0123, 0,      0     ],
                    [0,      0.0123, 0     ],
                    [0,      0,      0.0224]]
                )) # Inertial tensor (kg*m^2)
params["invI"] = torch.linalg.inv(params["IB"])
params["IRzz"] = 2.7e-5   # Rotor moment of inertia (kg*m^2)

params["Cd"]         = 0.1
params["kTh"]        = 1.076e-5 # thrust coeff (N/(rad/s)^2)  (1.18e-7 N/RPM^2)
params["kTo"]        = 1.632e-7 # torque coeff (Nm/(rad/s)^2)  (1.79e-9 Nm/RPM^2)
params["mixerFM"]    = utils.makeMixerFM(params) # Make mixer that calculated Thrust (F) and moments (M) as a function on motor speeds
params["mixerFMinv"] = torch.linalg.inv(params["mixerFM"])
params["minThr"]     = 0.1*4    # Minimum total thrust
params["maxThr"]     = 9.18*4   # Maximum total thrust
params["minWmotor"]  = 75       # Minimum motor rotation speed (rad/s)
params["maxWmotor"]  = 925      # Maximum motor rotation speed (rad/s)
params["tau"]        = 0.015    # Value for second order system for Motor dynamics
params["kp"]         = 1.0      # Value for second order system for Motor dynamics
params["damp"]       = 1.0      # Value for second order system for Motor dynamics

params["usePrecession"] = True  # Use precession or not
params["w_hover"] = 522.9847140714692
params["cmd_hover"] = [params["w_hover"]]*4
params["cmd_hover_pt"] = ptu.tensor(params["cmd_hover"])
params["maxCmd"] = 10
params["minCmd"] = -10
params["hover_thr"] = params["kTh"] * params["w_hover"] ** 2 * 4

params["default_init_state_list"] = [
    0,                  # x
    0,                  # y
    0,                  # z
    1,                  # q0
    0,                  # q1
    0,                  # q2
    0,                  # q3
    0,                  # xdot
    0,                  # ydot
    0,                  # zdot
    0,                  # p
    0,                  # q
    0,                  # r
    522.9847140714692,  # wM1
    522.9847140714692,  # wM2
    522.9847140714692,  # wM3
    522.9847140714692   # wM4
]

params["default_init_state_np"] = np.array(params["default_init_state_list"])

params["default_init_state_pt"] = ptu.from_numpy(params["default_init_state_np"])

params["state_ub"] = np.array([
    30, # np.inf,
    30, # np.inf,
    30, # np.inf,
    2*pi,
    2*pi,
    2*pi,
    2*pi,
    10,
    10,
    10,
    50,
    50,
    50,
    params["maxWmotor"],
    params["maxWmotor"],
    params["maxWmotor"],
    params["maxWmotor"],
])

params["state_lb"] = np.array([
    - 30, # np.inf,
    - 30, # np.inf,
    - 30, # np.inf,
    - 2*pi,
    - 2*pi,
    - 2*pi,
    - 2*pi,
    - 10,
    - 10,
    - 10,
    - 50,
    - 50,
    - 50,
    params["minWmotor"],
    params["minWmotor"],
    params["minWmotor"],
    params["minWmotor"],
])

params["rl_min"] = [
    - 2, # np.inf,
    - 2, # np.inf,
    - 2, # np.inf,
    - 0.2*pi,
    - 0.2*pi,
    - 0.2*pi,
    - 0.2*pi,
    - 0.1,
    - 0.1,
    - 0.1,
    - 0.05,
    - 0.05,
    - 0.05,
    params["minWmotor"],
    params["minWmotor"],
    params["minWmotor"],
    params["minWmotor"],
]

params["rl_max"] = [
    2, # np.inf,
    2, # np.inf,
    2, # np.inf,
    0.2*pi,
    0.2*pi,
    0.2*pi,
    0.2*pi,
    0.1,
    0.1,
    0.1,
    0.05,
    0.05,
    0.05,
    params["maxWmotor"],
    params["maxWmotor"],
    params["maxWmotor"],
    params["maxWmotor"],
]


params["ca_state_ub"] = ca.MX(params["state_ub"])
params["ca_state_lb"] = ca.MX(params["state_lb"])

# Normalization Parameters
# ------------------------

params["state_var"] = np.array([
    5, 5, 5, 
    1.66870635e-02,  1.80049500e-02, 1.90097617e-02, 3.94781769e-02, 
    6.79250264e-01,  5.99078863e-01, 5.46886039e-01, 
    1.51097522e+00, 1.48196943e+00,  2.04250634e-02, 
    6.02421510e+03, 6.00728172e+03, 5.79842870e+03,  6.09344182e+03
])

params["state_mean"] = np.array([
    0, 0, 0,  
    1, 0, 0, 0, 
    -3.56579433e-02, -2.08600217e-02,  5.04818729e-02, 
    -3.33658909e-02,  5.77660876e-02,  5.01574786e-04,  
    5.26444419e+02,  5.27135070e+02,  5.26640264e+02,  5.26382155e+02
])

params["rl_state_var"] = np.array([
    3.55325707e-01, 4.97226847e-01, 3.64094083e-01, 
    1.66870635e-02, 1.80049500e-02, 1.90097617e-02, 3.94781769e-02, 
    6.79250264e-01, 5.99078863e-01, 5.46886039e-01, 
    1.51097522e+00, 1.48196943e+00, 2.04250634e-02, 
    6.02421510e+03, 6.00728172e+03, 5.79842870e+03, 6.09344182e+03, 
    0.5, 0.5, 0.5
])

params["rl_state_mean"] = np.array([
    0, 0, 0,  
    9.52120032e-01,  -1.55759417e-03,  7.06277083e-03, -1.53357067e-02, 
    -3.56579433e-02, -2.08600217e-02,  5.04818729e-02, 
    -3.33658909e-02,  5.77660876e-02,  5.01574786e-04,  
    5.26444419e+02,  5.27135070e+02,  5.26640264e+02,  5.26382155e+02, 
    1, 1, 1
])

params["state_dot_mean"] = np.array([
    0,  0,  0, 0,
    0,  0, 0, 0,
    0,  0, 0, 0,
    0, 0,  0,  0,
    0
])

params["state_dot_var"] = np.array([
    1.05118748e-02, 4.84750726e+00, 4.86083715e+00, 1.53591744e-05,
    6.21927312e-05, 7.82977352e-02, 5.73502092e-05, 2.10500257e-01,
    2.31031707e-01, 1.24149857e-01, 1.24791653e-03, 3.39575201e-02,
    3.30781614e-06, 8.59376433e-01, 8.50812394e-01, 8.64048027e-01,
    8.64907141e-01
])

params["input_mean"] = np.array([
    0,0,0,0
])

params["input_var"] = np.array([0.9045, 0.8557, 0.8783, 0.8276])

params["w_mean"] = np.array([params["w_hover"]]*4)
params["w_var"] = np.array([50000]*4)

params["useIntegral"] = True

if __name__ == '__main__':
    # testing the variance of the states



    
    state = params["default_init_state_np"]
    state[1] = 5
    state[0] = 5
    state[2] = 5
    


    print('fin')



