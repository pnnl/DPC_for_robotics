import copy, os
import casadi as ca
import numpy as np
import torch
import utils.pytorch as ptu
import utils.quad
import utils.mujoco
import mujoco as mj

from typing import Dict

def get_quad_params():
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
    params["IB_pt"]   = ptu.tensor(
                        [[0.0123, 0,      0     ],
                        [0,      0.0123, 0     ],
                        [0,      0,      0.0224]]
                    ) # Inertial tensor (kg*m^2)
    params["IB"]   = np.array(
                        [[0.0123, 0,      0     ],
                        [0,      0.0123, 0     ],
                        [0,      0,      0.0224]]
                    ) # Inertial tensor (kg*m^2)
    params["invI"] = torch.linalg.inv(params["IB_pt"])
    params["IRzz"] = 2.7e-5   # Rotor moment of inertia (kg*m^2)

    params["Cd"]         = 0.1
    params["kTh"]        = 1.076e-5 # thrust coeff (N/(rad/s)^2)  (1.18e-7 N/RPM^2)
    params["kTo"]        = 1.632e-7 # torque coeff (Nm/(rad/s)^2)  (1.79e-9 Nm/RPM^2)
    params["mixerFM"]    = utils.quad.makeMixerFM.pytorch(params) # Make mixer that calculated Thrust (F) and moments (M) as a function on motor speeds
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
        2*np.pi,
        2*np.pi,
        2*np.pi,
        2*np.pi,
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
        - 2*np.pi,
        - 2*np.pi,
        - 2*np.pi,
        - 2*np.pi,
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
        - 0.2*np.pi,
        - 0.2*np.pi,
        - 0.2*np.pi,
        - 0.2*np.pi,
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
        0.2*np.pi,
        0.2*np.pi,
        0.2*np.pi,
        0.2*np.pi,
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

    return params

class state_dot:

    @staticmethod
    def numpy(state: np.ndarray, cmd: np.ndarray, params: Dict):

        q0 =    state[3]
        q1 =    state[4]
        q2 =    state[5]
        q3 =    state[6]
        xdot =  state[7]
        ydot =  state[8]
        zdot =  state[9]
        p =     state[10]
        q =     state[11]
        r =     state[12]
        wM1 =   state[13]
        wM2 =   state[14]
        wM3 =   state[15]
        wM4 =   state[16]

        wMotor = np.stack([wM1, wM2, wM3, wM4])
        wMotor = np.clip(wMotor, params["minWmotor"], params["maxWmotor"])
        thrust = params["kTh"] * wMotor ** 2
        torque = params["kTo"] * wMotor ** 2

        # Wind Model
        # ---------------------------
        velW, qW1, qW2 = [0]*3

        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        DynamicsDot = np.stack(
            [
                xdot,
                ydot,
                zdot,
                -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                (
                    params["Cd"]
                    * np.sign(velW * np.cos(qW1) * np.cos(qW2) - xdot)
                    * (velW * np.cos(qW1) * np.cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                )
                / params["mB"],
                (
                    params["Cd"]
                    * np.sign(velW * np.sin(qW1) * np.cos(qW2) - ydot)
                    * (velW * np.sin(qW1) * np.cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                )
                / params["mB"],
                (
                    -params["Cd"] * np.sign(velW * np.sin(qW2) + zdot) * (velW * np.sin(qW2) + zdot) ** 2
                    - (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + params["g"] * params["mB"]
                )
                / params["mB"],
                (
                    (params["IB"][1,1] - params["IB"][2,2]) * q * r
                    - params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * q
                    + (thrust[0] - thrust[1] - thrust[2] + thrust[3]) * params["dym"]
                )
                / params["IB"][0,0],  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (params["IB"][2,2] - params["IB"][0,0]) * p * r
                    + params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * p
                    + (thrust[0] + thrust[1] - thrust[2] - thrust[3]) * params["dxm"]
                )
                / params["IB"][1,1],  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((params["IB"][0,0] - params["IB"][1,1]) * p * q - torque[0] + torque[1] - torque[2] + torque[3]) / params["IB"][2,2],
            ]
        )

        # we must limit the actuator rotational rate
        omega_check_upper = state[13:] > params["maxWmotor"]
        omega_check_lower = state[13:] < params["minWmotor"]
        ActuatorsDot = cmd/params["IRzz"]
        ActuatorsDot[(omega_check_upper) | (omega_check_lower)] = 0

        # State Derivative Vector
        # ---------------------------
        if state.shape[0] == 1:
            return np.hstack([DynamicsDot.squeeze(), ActuatorsDot.squeeze()])
        else:
            return np.hstack([DynamicsDot.T, ActuatorsDot])

    @staticmethod
    def casadi(state: ca.MX, cmd: ca.MX, params: Dict):
        # formerly known as QuadcopterCA

        # Import params to numpy for CasADI
        # ---------------------------
        IB = params["IB"]
        IBxx = IB[0, 0]
        IByy = IB[1, 1]
        IBzz = IB[2, 2]

        # Unpack state tensor for readability
        # ---------------------------
        q0 =    state[3]
        q1 =    state[4]
        q2 =    state[5]
        q3 =    state[6]
        xdot =  state[7]
        ydot =  state[8]
        zdot =  state[9]
        p =     state[10]
        q =     state[11]
        r =     state[12]
        wM1 =   state[13]
        wM2 =   state[14]
        wM3 =   state[15]
        wM4 =   state[16]

        # a tiny bit more readable
        ThrM1 = params["kTh"] * wM1 ** 2
        ThrM2 = params["kTh"] * wM2 ** 2
        ThrM3 = params["kTh"] * wM3 ** 2
        ThrM4 = params["kTh"] * wM4 ** 2
        TorM1 = params["kTo"] * wM1 ** 2
        TorM2 = params["kTo"] * wM2 ** 2
        TorM3 = params["kTo"] * wM3 ** 2
        TorM4 = params["kTo"] * wM4 ** 2

        # Wind Model (zero in expectation)
        # ---------------------------
        velW, qW1, qW2 = 0, 0, 0

        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        DynamicsDot = ca.vertcat(
                xdot,
                ydot,
                zdot,
                -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                (
                    params["Cd"]
                    * ca.sign(velW * ca.cos(qW1) * ca.cos(qW2) - xdot)
                    * (velW * ca.cos(qW1) * ca.cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / params["mB"],
                (
                    params["Cd"]
                    * ca.sign(velW * ca.sin(qW1) * ca.cos(qW2) - ydot)
                    * (velW * ca.sin(qW1) * ca.cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / params["mB"],
                (
                    -params["Cd"] * ca.sign(velW * ca.sin(qW2) + zdot) * (velW * ca.sin(qW2) + zdot) ** 2
                    - (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + params["g"] * params["mB"]
                )
                / params["mB"],
                (
                    (IByy - IBzz) * q * r
                    - params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * q
                    + (ThrM1 - ThrM2 - ThrM3 + ThrM4) * params["dym"]
                )
                / IBxx,  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (IBzz - IBxx) * p * r
                    + params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * p
                    + (ThrM1 + ThrM2 - ThrM3 - ThrM4) * params["dxm"]
                )
                / IByy,  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((IBxx - IByy) * p * q - TorM1 + TorM2 - TorM3 + TorM4) / IBzz,
                cmd[0]/params["IRzz"], cmd[1]/params["IRzz"], cmd[2]/params["IRzz"], cmd[3]/params["IRzz"]
        )

        if DynamicsDot.shape[1] == 17:
            print('fin')

        # State Derivative Vector
        # ---------------------------
        return DynamicsDot

    @staticmethod
    def casadi_vectorized(state: ca.MX, cmd: ca.MX, params: Dict):

        # Import params to numpy for CasADI
        # ---------------------------
        IB = params["IB"]
        IBxx = IB[0, 0]
        IByy = IB[1, 1]
        IBzz = IB[2, 2]

        # Unpack state tensor for readability
        # ---------------------------
        q0 =    state[3,:]
        q1 =    state[4,:]
        q2 =    state[5,:]
        q3 =    state[6,:]
        xdot =  state[7,:]
        ydot =  state[8,:]
        zdot =  state[9,:]
        p =     state[10,:]
        q =     state[11,:]
        r =     state[12,:]
        wM1 =   state[13,:]
        wM2 =   state[14,:]
        wM3 =   state[15,:]
        wM4 =   state[16,:]

        # a tiny bit more readable
        ThrM1 = params["kTh"] * wM1 ** 2
        ThrM2 = params["kTh"] * wM2 ** 2
        ThrM3 = params["kTh"] * wM3 ** 2
        ThrM4 = params["kTh"] * wM4 ** 2
        TorM1 = params["kTo"] * wM1 ** 2
        TorM2 = params["kTo"] * wM2 ** 2
        TorM3 = params["kTo"] * wM3 ** 2
        TorM4 = params["kTo"] * wM4 ** 2

        # Wind Model (zero in expectation)
        # ---------------------------
        velW, qW1, qW2 = 0, 0, 0

        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        DynamicsDot = ca.vertcat(
                xdot,
                ydot,
                zdot,
                -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                (
                    params["Cd"]
                    * ca.sign(velW * ca.cos(qW1) * ca.cos(qW2) - xdot)
                    * (velW * ca.cos(qW1) * ca.cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / params["mB"],
                (
                    params["Cd"]
                    * ca.sign(velW * ca.sin(qW1) * ca.cos(qW2) - ydot)
                    * (velW * ca.sin(qW1) * ca.cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / params["mB"],
                (
                    -params["Cd"] * ca.sign(velW * ca.sin(qW2) + zdot) * (velW * ca.sin(qW2) + zdot) ** 2
                    - (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + params["g"] * params["mB"]
                )
                / params["mB"],
                (
                    (IByy - IBzz) * q * r
                    - params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * q
                    + (ThrM1 - ThrM2 - ThrM3 + ThrM4) * params["dym"]
                )
                / IBxx,  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (IBzz - IBxx) * p * r
                    + params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * p
                    + (ThrM1 + ThrM2 - ThrM3 - ThrM4) * params["dxm"]
                )
                / IByy,  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((IBxx - IByy) * p * q - TorM1 + TorM2 - TorM3 + TorM4) / IBzz,
                cmd[0,:]/params["IRzz"], cmd[1,:]/params["IRzz"], cmd[2,:]/params["IRzz"], cmd[3,:]/params["IRzz"]
        )

        # State Derivative Vector
        # ---------------------------
        return DynamicsDot
    
    @staticmethod
    def pytorch_vectorized(state: torch.Tensor, cmd: torch.Tensor, params: Dict):

        def ensure_2d(tensor):
            if len(tensor.shape) == 1:
                tensor = torch.unsqueeze(tensor, 0)
            return tensor
        
        state = ensure_2d(state)
        cmd = ensure_2d(cmd)

        # Unpack state tensor for readability
        # ---------------------------

        # this unbind method works on raw tensors, but not NM variables
        # x, y, z, q0, q1, q2, q3, xdot, ydot, zdot, p, q, r, wM1, wM2, wM3, wM4 = torch.unbind(state, dim=1)

        # try the state.unpack function that comes with the nm

        # this allows the NM to be compatible with 
        q0 =    state[:,3]
        q1 =    state[:,4]
        q2 =    state[:,5]
        q3 =    state[:,6]
        xdot =  state[:,7]
        ydot =  state[:,8]
        zdot =  state[:,9]
        p =     state[:,10]
        q =     state[:,11]
        r =     state[:,12]
        wM1 =   state[:,13]
        wM2 =   state[:,14]
        wM3 =   state[:,15]
        wM4 =   state[:,16]

        wMotor = torch.stack([wM1, wM2, wM3, wM4])
        wMotor = torch.clip(wMotor, params["minWmotor"], params["maxWmotor"])
        thrust = params["kTh"] * wMotor ** 2
        torque = params["kTo"] * wMotor ** 2

        # Wind Model
        # ---------------------------
        velW, qW1, qW2 = [torch.zeros(1, device=state.device)]*3

        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        DynamicsDot = torch.stack(
            [
                xdot,
                ydot,
                zdot,
                -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                (
                    params["Cd"]
                    * torch.sign(velW * torch.cos(qW1) * torch.cos(qW2) - xdot)
                    * (velW * torch.cos(qW1) * torch.cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                )
                / params["mB"],
                (
                    params["Cd"]
                    * torch.sign(velW * torch.sin(qW1) * torch.cos(qW2) - ydot)
                    * (velW * torch.sin(qW1) * torch.cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                )
                / params["mB"],
                (
                    -params["Cd"] * torch.sign(velW * torch.sin(qW2) + zdot) * (velW * torch.sin(qW2) + zdot) ** 2
                    - (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + params["g"] * params["mB"]
                )
                / params["mB"],
                (
                    (params["IB"][1,1] - params["IB"][2,2]) * q * r
                    - params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * q
                    + (thrust[0] - thrust[1] - thrust[2] + thrust[3]) * params["dym"]
                )
                / params["IB"][0,0],  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (params["IB"][2,2] - params["IB"][0,0]) * p * r
                    + params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * p
                    + (thrust[0] + thrust[1] - thrust[2] - thrust[3]) * params["dxm"]
                )
                / params["IB"][1,1],  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((params["IB"][0,0] - params["IB"][1,1]) * p * q - torque[0] + torque[1] - torque[2] + torque[3]) / params["IB"][2,2],
            ]
        )

        ActuatorsDot = cmd/params["IRzz"]

        # State Derivative Vector
        # ---------------------------
        if state.shape[0] == 1:
            return torch.hstack([DynamicsDot.squeeze(), ActuatorsDot.squeeze()])
        else:
            return torch.hstack([DynamicsDot.T, ActuatorsDot])

    @staticmethod
    def neuromancer(state: torch.Tensor, cmd: torch.Tensor, params: Dict):

        # this allows the NM to be compatible with 
        q0 =    state[...,3]
        q1 =    state[...,4]
        q2 =    state[...,5]
        q3 =    state[...,6]
        xdot =  state[...,7]
        ydot =  state[...,8]
        zdot =  state[...,9]
        p =     state[...,10]
        q =     state[...,11]
        r =     state[...,12]
        wM1 =   state[...,13]
        wM2 =   state[...,14]
        wM3 =   state[...,15]
        wM4 =   state[...,16]

        wMotor = torch.stack([wM1, wM2, wM3, wM4])
        wMotor = torch.clip(wMotor, params["minWmotor"], params["maxWmotor"])
        thrust = params["kTh"] * wMotor ** 2
        torque = params["kTo"] * wMotor ** 2

        # Wind Model
        # ---------------------------
        velW, qW1, qW2 = [torch.zeros(1, device=state.device)]*3

        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        DynamicsDot = torch.vstack(
            [
                xdot,
                ydot,
                zdot,
                -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                (
                    params["Cd"]
                    * torch.sign(velW * torch.cos(qW1) * torch.cos(qW2) - xdot)
                    * (velW * torch.cos(qW1) * torch.cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                )
                / params["mB"],
                (
                    params["Cd"]
                    * torch.sign(velW * torch.sin(qW1) * torch.cos(qW2) - ydot)
                    * (velW * torch.sin(qW1) * torch.cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                )
                / params["mB"],
                (
                    -params["Cd"] * torch.sign(velW * torch.sin(qW2) + zdot) * (velW * torch.sin(qW2) + zdot) ** 2
                    - (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + params["g"] * params["mB"]
                )
                / params["mB"],
                (
                    (params["IB"][1,1] - params["IB"][2,2]) * q * r
                    - params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * q
                    + (thrust[0] - thrust[1] - thrust[2] + thrust[3]) * params["dym"]
                )
                / params["IB"][0,0],  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (params["IB"][2,2] - params["IB"][0,0]) * p * r
                    + params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * p
                    + (thrust[0] + thrust[1] - thrust[2] - thrust[3]) * params["dxm"]
                )
                / params["IB"][1,1],  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((params["IB"][0,0] - params["IB"][1,1]) * p * q - torque[0] + torque[1] - torque[2] + torque[3]) / params["IB"][2,2],
            ]
        )

        # ensure the command is 2D
        if len(cmd.shape) == 3:
            cmd = cmd.squeeze(0)

        ActuatorsDot = cmd/params["IRzz"]
        return torch.hstack([DynamicsDot.T, ActuatorsDot])

class linmod:

    @staticmethod
    def pytorch(state: torch.Tensor, cmd: torch.Tensor, params: Dict):
        """
        This may look ridiculous, but it is imported from a symbolically derived linearisation in the 
        file jacobian_derivation.py
        """

        # Import State Vector
        # ---------------------------
        x = state[0]
        y = state[1]
        z = state[2]
        q0 = state[3]
        q1 = state[4]
        q2 = state[5]
        q3 = state[6]
        xdot = state[7]
        ydot = state[8]
        zdot = state[9]
        p = state[10]
        q = state[11]
        r = state[12]
        wM1 = state[13]
        wM2 = state[14]
        wM3 = state[15]
        wM4 = state[16]

        wMotor = torch.stack([wM1, wM2, wM3, wM4])
        wMotor = torch.clip(wMotor, params["minWmotor"], params["maxWmotor"])

        # Stochastic Terms in Expectation
        # ---------------------------
        velW = ptu.from_numpy(np.array(0))
        qW1 =  ptu.from_numpy(np.array(0))
        qW2 =  ptu.from_numpy(np.array(0))

        # Jacobian A matrix
        # ---------------------------
        # placeholder for 0 which is a tensor to allow for easy stacking
        _ = ptu.from_numpy(np.array(0))

        # the contribution of x, y, z to the state derivatives
        col123 = ptu.from_numpy(np.zeros([17, 3]))

        # contribution of the attitude quaternion to the state derivatives
        col4567 = torch.vstack(
            [
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, -0.5 * p, -0.5 * q, -0.5 * r]),
                torch.stack([0.5 * p, _, 0.5 * r, -0.5 * q]),
                torch.stack([0.5 * q, -0.5 * r, _, 0.5 * p]),
                torch.stack([0.5 * r, 0.5 * q, -0.5 * p, _]),
                torch.stack(
                    [
                        -2
                        * q2
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        -2
                        * q3
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        -2
                        * q0
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        -2
                        * q1
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                    ]
                ),
                torch.stack(
                    [
                        2
                        * q1
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        2
                        * q0
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        -2
                        * q3
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        -2
                        * q2
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                    ]
                ),
                torch.stack(
                    [
                        -2
                        * q0
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        2
                        * q1
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        2
                        * q2
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                        -2
                        * q3
                        * (
                            params["kTh"] * wM1 ** 2
                            + params["kTh"] * wM2 ** 2
                            + params["kTh"] * wM3 ** 2
                            + params["kTh"] * wM4 ** 2
                        )
                        / params["mB"],
                    ]
                ),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
            ]
        )

        # contribution of xdot, ydot, zdot to the state derivatives
        col8910 = torch.vstack(
            [
                torch.stack([_ + 1, _, _]),
                torch.stack([_, _ + 1, _]),
                torch.stack([_, _, _ + 1]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack(
                    [
                        (
                            params["Cd"]
                            * (-2 * velW * torch.cos(qW1) * torch.cos(qW2) + 2 * xdot)
                            * torch.sign(velW * torch.cos(qW1) * torch.cos(qW2) - xdot)
                        )
                        / params["mB"],
                        _,
                        _,
                    ]
                ),
                torch.stack(
                    [
                        _,
                        (
                            params["Cd"]
                            * (-2 * velW * torch.sin(qW1) * torch.cos(qW2) + 2 * ydot)
                            * torch.sign(velW * torch.sin(qW1) * torch.cos(qW2) - ydot)
                        )
                        / params["mB"],
                        _,
                    ]
                ),
                torch.stack(
                    [
                        _,
                        _,
                        (
                            -params["Cd"]
                            * (2 * velW * torch.sin(qW2) + 2 * zdot)
                            * torch.sign(velW * torch.sin(qW2) + zdot)
                        )
                        / params["mB"],
                    ]
                ),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
            ]
        )

        # contribution of p, q, r (body frame angular velocity) to the state derivatives
        cols11_12_13 = torch.vstack(
            [
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([-0.5 * q1, -0.5 * q2, -0.5 * q3]),
                torch.stack([0.5 * q0, -0.5 * q3, 0.5 * q2]),
                torch.stack([0.5 * q3, 0.5 * q0, -0.5 * q1]),
                torch.stack([-0.5 * q2, 0.5 * q1, 0.5 * q0]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack(
                    [
                        _,
                        (-params["IRzz"] * (wM1 - wM2 + wM3 - wM4) + r * (params["IB"][1,1] - params["IB"][2,2])) / params["IB"][0,0],
                        q * (params["IB"][1,1] - params["IB"][2,2]) / params["IB"][0,0],
                    ]
                ),
                torch.stack(
                    [
                        (params["IRzz"] * (wM1 - wM2 + wM3 - wM4) + r * (-params["IB"][0,0] + params["IB"][2,2])) / params["IB"][1,1],
                        _,
                        p * (-params["IB"][0,0] + params["IB"][2,2]) / params["IB"][1,1],
                    ]
                ),
                torch.stack([q * (params["IB"][0,0] - params["IB"][1,1]) / params["IB"][2,2], p * (params["IB"][0,0] - params["IB"][1,1]) / params["IB"][2,2], _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
                torch.stack([_, _, _]),
            ]
        )

        # contribution of the angular accelerations of the rotors to the state derivatives
        cols_14_15_16_17 = torch.vstack(
            [
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack(
                    [
                        -2 * params["kTh"] * wM1 * (2 * q0 * q2 + 2 * q1 * q3) / params["mB"],
                        -2 * params["kTh"] * wM2 * (2 * q0 * q2 + 2 * q1 * q3) / params["mB"],
                        -2 * params["kTh"] * wM3 * (2 * q0 * q2 + 2 * q1 * q3) / params["mB"],
                        -2 * params["kTh"] * wM4 * (2 * q0 * q2 + 2 * q1 * q3) / params["mB"],
                    ]
                ),
                torch.stack(
                    [
                        2 * params["kTh"] * wM1 * (2 * q0 * q1 - 2 * q2 * q3) / params["mB"],
                        2 * params["kTh"] * wM2 * (2 * q0 * q1 - 2 * q2 * q3) / params["mB"],
                        2 * params["kTh"] * wM3 * (2 * q0 * q1 - 2 * q2 * q3) / params["mB"],
                        2 * params["kTh"] * wM4 * (2 * q0 * q1 - 2 * q2 * q3) / params["mB"],
                    ]
                ),
                torch.stack(
                    [
                        -2 * params["kTh"] * wM1 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / params["mB"],
                        -2 * params["kTh"] * wM2 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / params["mB"],
                        -2 * params["kTh"] * wM3 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / params["mB"],
                        -2 * params["kTh"] * wM4 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / params["mB"],
                    ]
                ),
                torch.stack(
                    [
                        (-params["IRzz"] * q + 2 * params["dym"] * params["kTh"] * wM1) / params["IB"][0,0],
                        (params["IRzz"] * q - 2 * params["dym"] * params["kTh"] * wM2) / params["IB"][0,0],
                        (-params["IRzz"] * q - 2 * params["dym"] * params["kTh"] * wM3) / params["IB"][0,0],
                        (params["IRzz"] * q + 2 * params["dym"] * params["kTh"] * wM4) / params["IB"][0,0],
                    ]
                ),
                torch.stack(
                    [
                        (params["IRzz"] * p + 2 * params["dxm"] * params["kTh"] * wM1) / params["IB"][1,1],
                        (-params["IRzz"] * p + 2 * params["dxm"] * params["kTh"] * wM2) / params["IB"][1,1],
                        (params["IRzz"] * p - 2 * params["dxm"] * params["kTh"] * wM3) / params["IB"][1,1],
                        (-params["IRzz"] * p - 2 * params["dxm"] * params["kTh"] * wM4) / params["IB"][1,1],
                    ]
                ),
                torch.stack(
                    [
                        -2 * params["kTo"] * wM1 / params["IB"][2,2],
                        2 * params["kTo"] * wM2 / params["IB"][2,2],
                        -2 * params["kTo"] * wM3 / params["IB"][2,2],
                        2 * params["kTo"] * wM4 / params["IB"][2,2],
                    ]
                ),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
            ]
        )

        A = torch.hstack([col123, col4567, col8910, cols11_12_13, cols_14_15_16_17])

        # Jacobian B matrix
        # ---------------------------

        # contribution of the input torques to the state derivatives
        B = torch.vstack(
            [
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_, _, _, _]),
                torch.stack([_ + 1 / params["IRzz"], _, _, _]),
                torch.stack([_, _ + 1 / params["IRzz"], _, _]),
                torch.stack([_, _, _ + 1 / params["IRzz"], _]),
                torch.stack([_, _, _, _ + 1 / params["IRzz"]]),
            ]
        )

        return A, B

class mujoco_quad:

    """
    The rather annoying thing about mujoco is it is stateful, it holds a state
    within it, which I must get and set and reset when necessary. That being said
    I will not hold an internal memory in this class of the state on the python side
    for simplicity.

    I do however need to hold a state of the omegas, as mujoco does not simulate them.
    """

    def __init__(
            self,
            state,
            quad_params,
            Ti, Tf, Ts,
            integrator,
            xml_path='mujoco_quad.xml',
            render='matplotlib' # 'matplotlib', 'mujoco'    
        ):

        self.integrator = integrator
        self.Ts = Ts
        self.Ti = Ti
        self.Tf = Tf
        self.quad_params = quad_params # used to do a lot of the instantiating
        self.t = Ti
        self.render = render
        self.xml_path = xml_path

        self.omegas = copy.deepcopy(state[13:17])

        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath

        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(xml_path, )  # MuJoCo model
        self.model.opt.timestep = self.Ts

        assert self.Ts == self.model.opt.timestep # should be 0.01
        # assert self.model.opt.integrator == 0 # 0 == semi implicit euler, closest to explicit euler mj offers
        # mjdata constains the state and quantities that depend on it.
        if self.integrator == 'euler':
            # this is NOT explicit euler this is semi-implicit euler
            self.model.opt.integrator = 0
        elif self.integrator == 'rk4':
            self.model.opt.integrator = 1
        self.data = mj.MjData(self.model)

        # Make renderer, render and show the pixels
        if render == 'mujoco':
            self.renderer = mj.Renderer(model=self.model, height=720, width=1280)
            # self.data.cam_xpos = np.array([[1,2,3]])
            self.model.cam_pos0 = np.array([[1,2,3]])
            self.model.cam_pos = np.array([[1,2,3]])

        mj.mj_resetData(self.model, self.data)  # Reset state and time.
        self.data.ctrl = [self.quad_params["kTh"] * self.quad_params["w_hover"] ** 2] * 4 # kTh * w_hover ** 2 = 2.943

        self.current_cmd = np.zeros(4)

        # set initial conditions, track with state attribute for convenience
        self.set_state(copy.deepcopy(state))

        # mujoco operates on numpy arrays not tensors
        self.state = copy.deepcopy(state)

    def __call__(self, state, cmd):
        # designed for 1-D use with Neuromancer
        return ptu.from_numpy(self.step(ptu.to_numpy(cmd.squeeze(0)))).unsqueeze(0)

    def step(
            self, 
            cmd: np.ndarray,
        ):
        assert isinstance(cmd, np.ndarray), "cmd should be a np.ndarray for mujoco sim"
        self.current_cmd = cmd

        # translate omegas to thrust (mj input)
        thr = self.quad_params["kTh"] * self.omegas ** 2
        self.data.ctrl = thr.tolist()

        # update mujoco and actuators with EULER
        mj.mj_step(self.model, self.data)
        self.omegas += cmd/self.quad_params["IRzz"] * self.Ts

        # retrieve time for the environment
        self.t = self.data.time

        self.state = self.get_state()

        return self.state

    def get_state(self):

        return utils.mujoco.mj_get_state(self.data, self.omegas)

    def set_state(self, state):

        # convert state to mujoco compatible 
        qpos, qvel = utils.mujoco.state2qpv(state)

        # apply
        self.data.qpos = qpos
        self.data.qvel = qvel

        # handle the rotors (omegas) and state save separately
        self.omegas = copy.deepcopy(state.squeeze()[13:17])
        self.state = copy.deepcopy(state)

    def reset(self, state):

        print('performing mujoco reset')

        self.omegas = np.array([self.quad_params["w_hover"]]*4)
        self.set_state(copy.deepcopy(state))

        # added during sysID phase
        self.t = self.Ti

if __name__ == "__main__":

    # an example simulation and animation 
    from utils.integrate import euler, RK4
    from utils.quad import Animator

    ptu.init_dtype()
    ptu.init_gpu()

    def test_state_dot():
        quad_params = get_quad_params()
        state = quad_params["default_init_state_pt"]
        input = ptu.tensor([0.0001,0.0,0.0,0.0])

        Ti, Tf, Ts = 0.0, 3.0, 0.1
        memory = {'state': [ptu.to_numpy(state)], 'input': [ptu.to_numpy(input)]}
        times = np.arange(Ti, Tf, Ts)
        for t in times:

            state = RK4.time_invariant.pytorch(state_dot.pytorch_vectorized, state, input, Ts, quad_params)

            memory['state'].append(ptu.to_numpy(state))
            memory['input'].append(ptu.to_numpy(input))

        memory['state'] = np.vstack(memory['state'])
        memory['input'] = np.vstack(memory['input'])

        animator = Animator(memory['state'], times, memory['state'], max_frames=10, save_path='data')
        animator.animate()

    def test_mujoco_quad():
        quad_params = get_quad_params()
        state = quad_params["default_init_state_np"]
        input = np.array([0.0001,0.0,0.0,0.0])
        Ti, Tf, Ts = 0.0, 3.0, 0.1

        mj_quad = mujoco_quad(state=state, quad_params=quad_params, Ti=Ti, Tf=Tf, Ts=Ts, integrator='euler')

        memory = {'state': [state], 'input': [input]}
        times = np.arange(Ti, Tf, Ts)
        for t in times:

            state = mj_quad.step(input)

            memory['state'].append(state)
            memory['input'].append(input)

        memory['state'] = np.vstack(memory['state'])
        memory['input'] = np.vstack(memory['input'])

        # needs to be followed by a reset if we are to repeat the simulation
        mj_quad.reset(quad_params["default_init_state_np"])

        animator = Animator(memory['state'], times, memory['state'], max_frames=10, save_path='data')
        animator.animate()

    test_mujoco_quad()

    print('fin')