#

# -*- coding: utf-8 -*-
"""
original author, link to original: https://github.com/bobzwik/Quadcopter_SimCon
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

"""
edits:
    - Dynamics ported to PyTorch for differentiability
    - Rotor actuators 2nd order -> 1st order
        - 21 states -> 17 states
    - Jacobian formed and validated
        - SymPy derived jacobian, validated through generating the functorch.jacrev
          jacobian using the nonlinear dynamics and validating consistency with
          equations.
    - Minor changes to reduce number of files/complexity
"""

import casadi as ca
import torch
from torch import sin, cos, tan, pi, sign
import numpy as np

import utils
from utils import pytorch_utils as ptu

# This quadcopter class contains no state, only simulation parameters and function
class Quadcopter:
    def __init__(self, nm=False):

        # Whether or not to expect neuromancer variables rather than torch tensors for state and command
        self.nm = nm

        # Quad Params
        # ---------------------------
        params = {}
        params["mB"]   = 1.2       # mass (kg)
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
        params["maxCmd"] = 100
        params["minCmd"] = -100

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
            5,
            5,
            5,
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
            - 5,
            - 5,
            - 5,
            params["minWmotor"],
            params["minWmotor"],
            params["minWmotor"],
            params["minWmotor"],
        ])

        params["ca_state_ub"] = ca.MX(params["state_ub"])
        params["ca_state_lb"] = ca.MX(params["state_lb"])
        self.params = params

    def state_dot(self, state: torch.Tensor, cmd: torch.Tensor):

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
        wMotor = torch.clip(wMotor, self.params["minWmotor"], self.params["maxWmotor"])
        thrust = self.params["kTh"] * wMotor ** 2
        torque = self.params["kTo"] * wMotor ** 2

        # Wind Model
        # ---------------------------
        velW, qW1, qW2 = [ptu.from_numpy(np.array(0))]*3

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
                    self.params["Cd"]
                    * sign(velW * cos(qW1) * cos(qW2) - xdot)
                    * (velW * cos(qW1) * cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                )
                / self.params["mB"],
                (
                    self.params["Cd"]
                    * sign(velW * sin(qW1) * cos(qW2) - ydot)
                    * (velW * sin(qW1) * cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                )
                / self.params["mB"],
                (
                    -self.params["Cd"] * sign(velW * sin(qW2) + zdot) * (velW * sin(qW2) + zdot) ** 2
                    - (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + self.params["g"] * self.params["mB"]
                )
                / self.params["mB"],
                (
                    (self.params["IB"][1,1] - self.params["IB"][2,2]) * q * r
                    - self.params["usePrecession"] * self.params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * q
                    + (thrust[0] - thrust[1] - thrust[2] + thrust[3]) * self.params["dym"]
                )
                / self.params["IB"][0,0],  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (self.params["IB"][2,2] - self.params["IB"][0,0]) * p * r
                    + self.params["usePrecession"] * self.params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * p
                    + (thrust[0] + thrust[1] - thrust[2] - thrust[3]) * self.params["dxm"]
                )
                / self.params["IB"][1,1],  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((self.params["IB"][0,0] - self.params["IB"][1,1]) * p * q - torque[0] + torque[1] - torque[2] + torque[3]) / self.params["IB"][2,2],
            ]
        )

        ActuatorsDot = cmd/self.params["IRzz"]

        # State Derivative Vector
        # ---------------------------
        if state.shape[0] == 1:
            return torch.hstack([DynamicsDot.squeeze(), ActuatorsDot.squeeze()])
        else:
            return torch.hstack([DynamicsDot.T, ActuatorsDot])
    
    def casadi_state_dot(self, state: ca.MX, cmd: ca.MX):

        # Import params to numpy for CasADI
        # ---------------------------
        IB = self.params["IB"]
        IBxx = ptu.to_numpy(IB[0, 0])
        IByy = ptu.to_numpy(IB[1, 1])
        IBzz = ptu.to_numpy(IB[2, 2])

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
        ThrM1 = self.params["kTh"] * wM1 ** 2
        ThrM2 = self.params["kTh"] * wM2 ** 2
        ThrM3 = self.params["kTh"] * wM3 ** 2
        ThrM4 = self.params["kTh"] * wM4 ** 2
        TorM1 = self.params["kTo"] * wM1 ** 2
        TorM2 = self.params["kTo"] * wM2 ** 2
        TorM3 = self.params["kTo"] * wM3 ** 2
        TorM4 = self.params["kTo"] * wM4 ** 2

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
                    self.params["Cd"]
                    * ca.sign(velW * ca.cos(qW1) * ca.cos(qW2) - xdot)
                    * (velW * ca.cos(qW1) * ca.cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / self.params["mB"],
                (
                    self.params["Cd"]
                    * ca.sign(velW * ca.sin(qW1) * ca.cos(qW2) - ydot)
                    * (velW * ca.sin(qW1) * ca.cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / self.params["mB"],
                (
                    -self.params["Cd"] * ca.sign(velW * ca.sin(qW2) + zdot) * (velW * ca.sin(qW2) + zdot) ** 2
                    - (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + self.params["g"] * self.params["mB"]
                )
                / self.params["mB"],
                (
                    (IByy - IBzz) * q * r
                    - self.params["usePrecession"] * self.params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * q
                    + (ThrM1 - ThrM2 - ThrM3 + ThrM4) * self.params["dym"]
                )
                / IBxx,  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (IBzz - IBxx) * p * r
                    + self.params["usePrecession"] * self.params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * p
                    + (ThrM1 + ThrM2 - ThrM3 - ThrM4) * self.params["dxm"]
                )
                / IByy,  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((IBxx - IByy) * p * q - TorM1 + TorM2 - TorM3 + TorM4) / IBzz,
                cmd[0,:]/self.params["IRzz"], cmd[1,:]/self.params["IRzz"], cmd[2,:]/self.params["IRzz"], cmd[3,:]/self.params["IRzz"]
        )

        # State Derivative Vector
        # ---------------------------
        return DynamicsDot

    def linmod(self, state): # VALIDATED

        """This may look ridiculous, but it is imported from a symbolically derived linearisation in the 
        file jacobian_derivation.py"""

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
        wMotor = torch.clip(wMotor, self.params["minWmotor"], self.params["maxWmotor"])

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
                            self.params["kTh"] * wM1 ** 2
                            + self.params["kTh"] * wM2 ** 2
                            + self.params["kTh"] * wM3 ** 2
                            + self.params["kTh"] * wM4 ** 2
                        )
                        / self.params["mB"],
                        -2
                        * q3
                        * (
                            self.params["kTh"] * wM1 ** 2
                            + self.params["kTh"] * wM2 ** 2
                            + self.params["kTh"] * wM3 ** 2
                            + self.params["kTh"] * wM4 ** 2
                        )
                        / self.params["mB"],
                        -2
                        * q0
                        * (
                            self.params["kTh"] * wM1 ** 2
                            + self.params["kTh"] * wM2 ** 2
                            + self.params["kTh"] * wM3 ** 2
                            + self.params["kTh"] * wM4 ** 2
                        )
                        / self.params["mB"],
                        -2
                        * q1
                        * (
                            self.params["kTh"] * wM1 ** 2
                            + self.params["kTh"] * wM2 ** 2
                            + self.params["kTh"] * wM3 ** 2
                            + self.params["kTh"] * wM4 ** 2
                        )
                        / self.params["mB"],
                    ]
                ),
                torch.stack(
                    [
                        2
                        * q1
                        * (
                            self.params["kTh"] * wM1 ** 2
                            + self.params["kTh"] * wM2 ** 2
                            + self.params["kTh"] * wM3 ** 2
                            + self.params["kTh"] * wM4 ** 2
                        )
                        / self.params["mB"],
                        2
                        * q0
                        * (
                            self.params["kTh"] * wM1 ** 2
                            + self.params["kTh"] * wM2 ** 2
                            + self.params["kTh"] * wM3 ** 2
                            + self.params["kTh"] * wM4 ** 2
                        )
                        / self.params["mB"],
                        -2
                        * q3
                        * (
                            self.params["kTh"] * wM1 ** 2
                            + self.params["kTh"] * wM2 ** 2
                            + self.params["kTh"] * wM3 ** 2
                            + self.params["kTh"] * wM4 ** 2
                        )
                        / self.params["mB"],
                        -2
                        * q2
                        * (
                            self.params["kTh"] * wM1 ** 2
                            + self.params["kTh"] * wM2 ** 2
                            + self.params["kTh"] * wM3 ** 2
                            + self.params["kTh"] * wM4 ** 2
                        )
                        / self.params["mB"],
                    ]
                ),
                torch.stack(
                    [
                        -2
                        * q0
                        * (
                            self.params["kTh"] * wM1 ** 2
                            + self.params["kTh"] * wM2 ** 2
                            + self.params["kTh"] * wM3 ** 2
                            + self.params["kTh"] * wM4 ** 2
                        )
                        / self.params["mB"],
                        2
                        * q1
                        * (
                            self.params["kTh"] * wM1 ** 2
                            + self.params["kTh"] * wM2 ** 2
                            + self.params["kTh"] * wM3 ** 2
                            + self.params["kTh"] * wM4 ** 2
                        )
                        / self.params["mB"],
                        2
                        * q2
                        * (
                            self.params["kTh"] * wM1 ** 2
                            + self.params["kTh"] * wM2 ** 2
                            + self.params["kTh"] * wM3 ** 2
                            + self.params["kTh"] * wM4 ** 2
                        )
                        / self.params["mB"],
                        -2
                        * q3
                        * (
                            self.params["kTh"] * wM1 ** 2
                            + self.params["kTh"] * wM2 ** 2
                            + self.params["kTh"] * wM3 ** 2
                            + self.params["kTh"] * wM4 ** 2
                        )
                        / self.params["mB"],
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
                            self.params["Cd"]
                            * (-2 * velW * cos(qW1) * cos(qW2) + 2 * xdot)
                            * sign(velW * cos(qW1) * cos(qW2) - xdot)
                        )
                        / self.params["mB"],
                        _,
                        _,
                    ]
                ),
                torch.stack(
                    [
                        _,
                        (
                            self.params["Cd"]
                            * (-2 * velW * sin(qW1) * cos(qW2) + 2 * ydot)
                            * sign(velW * sin(qW1) * cos(qW2) - ydot)
                        )
                        / self.params["mB"],
                        _,
                    ]
                ),
                torch.stack(
                    [
                        _,
                        _,
                        (
                            -self.params["Cd"]
                            * (2 * velW * sin(qW2) + 2 * zdot)
                            * sign(velW * sin(qW2) + zdot)
                        )
                        / self.params["mB"],
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
                        (-self.params["IRzz"] * (wM1 - wM2 + wM3 - wM4) + r * (self.params["IB"][1,1] - self.params["IB"][2,2])) / self.params["IB"][0,0],
                        q * (self.params["IB"][1,1] - self.params["IB"][2,2]) / self.params["IB"][0,0],
                    ]
                ),
                torch.stack(
                    [
                        (self.params["IRzz"] * (wM1 - wM2 + wM3 - wM4) + r * (-self.params["IB"][0,0] + self.params["IB"][2,2])) / self.params["IB"][1,1],
                        _,
                        p * (-self.params["IB"][0,0] + self.params["IB"][2,2]) / self.params["IB"][1,1],
                    ]
                ),
                torch.stack([q * (self.params["IB"][0,0] - self.params["IB"][1,1]) / self.params["IB"][2,2], p * (self.params["IB"][0,0] - self.params["IB"][1,1]) / self.params["IB"][2,2], _]),
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
                        -2 * self.params["kTh"] * wM1 * (2 * q0 * q2 + 2 * q1 * q3) / self.params["mB"],
                        -2 * self.params["kTh"] * wM2 * (2 * q0 * q2 + 2 * q1 * q3) / self.params["mB"],
                        -2 * self.params["kTh"] * wM3 * (2 * q0 * q2 + 2 * q1 * q3) / self.params["mB"],
                        -2 * self.params["kTh"] * wM4 * (2 * q0 * q2 + 2 * q1 * q3) / self.params["mB"],
                    ]
                ),
                torch.stack(
                    [
                        2 * self.params["kTh"] * wM1 * (2 * q0 * q1 - 2 * q2 * q3) / self.params["mB"],
                        2 * self.params["kTh"] * wM2 * (2 * q0 * q1 - 2 * q2 * q3) / self.params["mB"],
                        2 * self.params["kTh"] * wM3 * (2 * q0 * q1 - 2 * q2 * q3) / self.params["mB"],
                        2 * self.params["kTh"] * wM4 * (2 * q0 * q1 - 2 * q2 * q3) / self.params["mB"],
                    ]
                ),
                torch.stack(
                    [
                        -2 * self.params["kTh"] * wM1 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / self.params["mB"],
                        -2 * self.params["kTh"] * wM2 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / self.params["mB"],
                        -2 * self.params["kTh"] * wM3 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / self.params["mB"],
                        -2 * self.params["kTh"] * wM4 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / self.params["mB"],
                    ]
                ),
                torch.stack(
                    [
                        (-self.params["IRzz"] * q + 2 * self.params["dym"] * self.params["kTh"] * wM1) / self.params["IB"][0,0],
                        (self.params["IRzz"] * q - 2 * self.params["dym"] * self.params["kTh"] * wM2) / self.params["IB"][0,0],
                        (-self.params["IRzz"] * q - 2 * self.params["dym"] * self.params["kTh"] * wM3) / self.params["IB"][0,0],
                        (self.params["IRzz"] * q + 2 * self.params["dym"] * self.params["kTh"] * wM4) / self.params["IB"][0,0],
                    ]
                ),
                torch.stack(
                    [
                        (self.params["IRzz"] * p + 2 * self.params["dxm"] * self.params["kTh"] * wM1) / self.params["IB"][1,1],
                        (-self.params["IRzz"] * p + 2 * self.params["dxm"] * self.params["kTh"] * wM2) / self.params["IB"][1,1],
                        (self.params["IRzz"] * p - 2 * self.params["dxm"] * self.params["kTh"] * wM3) / self.params["IB"][1,1],
                        (-self.params["IRzz"] * p - 2 * self.params["dxm"] * self.params["kTh"] * wM4) / self.params["IB"][1,1],
                    ]
                ),
                torch.stack(
                    [
                        -2 * self.params["kTo"] * wM1 / self.params["IB"][2,2],
                        2 * self.params["kTo"] * wM2 / self.params["IB"][2,2],
                        -2 * self.params["kTo"] * wM3 / self.params["IB"][2,2],
                        2 * self.params["kTo"] * wM4 / self.params["IB"][2,2],
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
                torch.stack([_ + 1 / self.params["IRzz"], _, _, _]),
                torch.stack([_, _ + 1 / self.params["IRzz"], _, _]),
                torch.stack([_, _, _ + 1 / self.params["IRzz"], _]),
                torch.stack([_, _, _, _ + 1 / self.params["IRzz"]]),
            ]
        )

        return A, B

if __name__ == "__main__":
    # running a main() inside the __name__ __main__ keeps vscode outline variables contained
    def main():
        # quick check to validate it is equivalent to old quad
        from quad import Quadcopter
        from quad_refactor import Quadcopter
        import copy

        # give us a baseline
        quad = Quadcopter()
        wind = utils.windModel.Wind('None', 2.0, 90, -15)

        # give us the functions with the quadcopter parameters
        quadrf = Quadcopter()

        # initial conditions
        state=ptu.from_numpy(np.array([
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
        ]) + 0.1)*(ptu.from_numpy(np.random.rand(17))-0.5)

        cmd = ptu.from_numpy(np.random.rand(4))

        dt = 0.1

        # sget original update
        quad.state = copy.deepcopy(state)
        new_state = quad.update(cmd, wind, 0, dt)

        new_state_rf = state + quadrf.state_dot(copy.deepcopy(state), cmd) * dt
        
        print((new_state - new_state_rf).abs().max())
        

    main()
