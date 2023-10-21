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

"""
notes:
    - This document could be rewritten to have dramatically less/better code, but it is 
      currently CORRECT, and it will take time to PROVE that a rewritten code is
      also correct.
"""

import torch
from torch import sin, cos, tan, pi, sign
import numpy as np

import utils
from utils import pytorch_utils as ptu

import casadi as ca
from torch.linalg import inv

deg2rad = pi / 180.0

def sys_params():
    mB  = 1.2       # mass (kg)
    g   = 9.81      # gravity (m/s/s)
    dxm = 0.16      # arm length (m)
    dym = 0.16      # arm length (m)
    dzm = 0.05      # motor height (m)
    IB  = np.array([[0.0123, 0,      0     ],
                    [0,      0.0123, 0     ],
                    [0,      0,      0.0224]]) # Inertial tensor (kg*m^2)
    IB = ptu.from_numpy(IB)
    IRzz = 2.7e-5   # Rotor moment of inertia (kg*m^2)


    params = {}
    params["mB"]   = mB
    params["g"]    = g
    params["dxm"]  = dxm
    params["dym"]  = dym
    params["dzm"]  = dzm
    params["IB"]   = IB
    params["invI"] = inv(IB)
    params["IRzz"] = IRzz

    params["Cd"]         = 0.1
    params["kTh"]        = 1.076e-5 # thrust coeff (N/(rad/s)^2)  (1.18e-7 N/RPM^2)
    params["kTo"]        = 1.632e-7 # torque coeff (Nm/(rad/s)^2)  (1.79e-9 Nm/RPM^2)
    params["mixerFM"]    = makeMixerFM(params) # Make mixer that calculated Thrust (F) and moments (M) as a function on motor speeds
    params["mixerFMinv"] = inv(params["mixerFM"])
    params["minThr"]     = 0.1*4    # Minimum total thrust
    params["maxThr"]     = 9.18*4   # Maximum total thrust
    params["minWmotor"]  = 75       # Minimum motor rotation speed (rad/s)
    params["maxWmotor"]  = 925      # Maximum motor rotation speed (rad/s)
    params["tau"]        = 0.015    # Value for second order system for Motor dynamics
    params["kp"]         = 1.0      # Value for second order system for Motor dynamics
    params["damp"]       = 1.0      # Value for second order system for Motor dynamics
    
    params["motorc1"]    = 8.49     # w (rad/s) = cmd*c1 + c0 (cmd in %)
    params["motorc0"]    = 74.7
    params["motordeadband"] = 1   
    params["usePrecession"] = True  # Use precession or not
    params["w_hover"] = 522.9847140714692
    params["useIntegral"] = True
    # params["ifexpo"] = bool(False)
    # if params["ifexpo"]:
    #     params["maxCmd"] = 100      # cmd (%) min and max
    #     params["minCmd"] = 0.01
    # else:
    #     params["maxCmd"] = 100
    #     params["minCmd"] = 1

    casadi_state_upper_bound = [
        ca.inf,
        ca.inf,
        ca.inf,
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
    ]

    casadi_state_lower_bound = [
        - ca.inf,
        - ca.inf,
        - ca.inf,
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
    ]

    params['ub'] = casadi_state_upper_bound
    params['lb'] = casadi_state_lower_bound


    casadi_constraints = {
        'upper': ca.MX(casadi_state_upper_bound),
        'lower': ca.MX(casadi_state_lower_bound) 
    }
    
    return params, casadi_constraints, (casadi_state_lower_bound, casadi_state_upper_bound)

def makeMixerFM(params):
    dxm = params["dxm"]
    dym = params["dym"]
    kTh = params["kTh"]
    kTo = params["kTo"] 

    # Motor 1 is front left, then clockwise numbering.
    # A mixer like this one allows to find the exact RPM of each motor 
    # given a desired thrust and desired moments.
    # Inspiration for this mixer (or coefficient matrix) and how it is used : 
    # https://link.springer.com/article/10.1007/s13369-017-2433-2 (https://sci-hub.tw/10.1007/s13369-017-2433-2)
    mixerFM = np.array([[    kTh,      kTh,      kTh,      kTh],
                        [dym*kTh, -dym*kTh,  -dym*kTh, dym*kTh],
                        [dxm*kTh,  dxm*kTh, -dxm*kTh, -dxm*kTh],
                        [   -kTo,      kTo,     -kTo,      kTo]])
    mixerFM = ptu.from_numpy(mixerFM)

    
    return mixerFM


def init_state():
    
    x0     = 0.  # m
    y0     = 0.  # m
    z0     = 0.  # m
    phi0   = 0.  # rad
    theta0 = 0.  # rad
    psi0   = 0.  # rad

    quat = utils.YPRToQuat(psi0, theta0, phi0)

    s = ptu.from_numpy(np.zeros(17))
    s[0]  = x0       # x
    s[1]  = y0       # y
    s[2]  = z0       # z
    s[3]  = quat[0]  # q0
    s[4]  = quat[1]  # q1
    s[5]  = quat[2]  # q2
    s[6]  = quat[3]  # q3
    s[7]  = 0.       # xdot
    s[8]  = 0.       # ydot
    s[9]  = 0.       # zdot
    s[10] = 0.       # p
    s[11] = 0.       # q
    s[12] = 0.       # r

    # rotational rate for hover
    s[13] = 522.9847140714692 # wM1 
    s[14] = 522.9847140714692 # wM2
    s[15] = 522.9847140714692 # wM3
    s[16] = 522.9847140714692 # wM4
    
    return s

class Quadcopter:
    def __init__(self):
        # Quad Params
        # ---------------------------
        self.params, self.casadi_constraints, self.raw_constraints = sys_params()
        self.thr = ptu.from_numpy(np.ones(4))
        self.tor = ptu.from_numpy(np.ones(4))

        # Initial State
        # ---------------------------
        self.state = init_state()

        self.pos = self.state[0:3]
        self.quat = self.state[3:7]
        self.vel = self.state[7:10]
        self.omega = self.state[10:13]
        self.wMotor = torch.stack(
            [self.state[13], self.state[14], self.state[15], self.state[16]]
        )  # stacking maintains device, dtype, graph continuity
        self.vel_dot = ptu.from_numpy(np.zeros(3))
        self.omega_dot = ptu.from_numpy(np.zeros(3))
        self.acc = ptu.from_numpy(np.zeros(3))

        self.extended_state()
        self.forces()



    def extended_state(self):
        # Rotation Matrix of current state (Direct Cosine Matrix)
        self.dcm = utils.quat2Dcm(self.quat)

        # Euler angles of current state
        YPR = utils.quatToYPR_ZYX(self.quat)
        self.euler = torch.flip(
            YPR, (0,)
        )  # flip YPR so that euler state = phi, theta, psi
        self.psi = YPR[0]
        self.theta = YPR[1]
        self.phi = YPR[2]

    def forces(self):
        # Rotor thrusts and torques
        self.thr = self.params["kTh"] * self.wMotor ** 2
        self.tor = self.params["kTo"] * self.wMotor ** 2

    def state_dot(self, state, cmd):
        # Import Params
        # ---------------------------
        mB = self.params["mB"]
        g = self.params["g"]
        dxm = self.params["dxm"]
        dym = self.params["dym"]
        IB = self.params["IB"]
        IBxx = IB[0, 0]
        IByy = IB[1, 1]
        IBzz = IB[2, 2]
        Cd = self.params["Cd"]

        kTh = self.params["kTh"]
        kTo = self.params["kTo"]
        tau = self.params["tau"]
        kp = self.params["kp"]
        damp = self.params["damp"]
        minWmotor = self.params["minWmotor"]
        maxWmotor = self.params["maxWmotor"]

        IRzz = self.params["IRzz"]
        if self.params["usePrecession"]:
            uP = 1
        else:
            uP = 0

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
        wMotor = torch.clip(wMotor, minWmotor, maxWmotor)
        thrust = kTh * wMotor * wMotor
        torque = kTo * wMotor * wMotor

        ThrM1 = thrust[0]
        ThrM2 = thrust[1]
        ThrM3 = thrust[2]
        ThrM4 = thrust[3]
        TorM1 = torque[0]
        TorM2 = torque[1]
        TorM3 = torque[2]
        TorM4 = torque[3]

        # Wind Model
        # ---------------------------
        # [velW, qW1, qW2] = wind.randomWind(t)
        velW, qW1, qW2 = 0, 0, 0
        # velW = 0
        velW = ptu.from_numpy(np.array(velW))
        qW1 = ptu.from_numpy(np.array(qW1))
        qW2 = ptu.from_numpy(np.array(qW2))


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
                    Cd
                    * sign(velW * cos(qW1) * cos(qW2) - xdot)
                    * (velW * cos(qW1) * cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / mB,
                (
                    Cd
                    * sign(velW * sin(qW1) * cos(qW2) - ydot)
                    * (velW * sin(qW1) * cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / mB,
                (
                    -Cd * sign(velW * sin(qW2) + zdot) * (velW * sin(qW2) + zdot) ** 2
                    - (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + g * mB
                )
                / mB,
                (
                    (IByy - IBzz) * q * r
                    - uP * IRzz * (wM1 - wM2 + wM3 - wM4) * q
                    + (ThrM1 - ThrM2 - ThrM3 + ThrM4) * dym
                )
                / IBxx,  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (IBzz - IBxx) * p * r
                    + uP * IRzz * (wM1 - wM2 + wM3 - wM4) * p
                    + (ThrM1 + ThrM2 - ThrM3 - ThrM4) * dxm
                )
                / IByy,  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((IBxx - IByy) * p * q - TorM1 + TorM2 - TorM3 + TorM4) / IBzz,
            ]
        )

        # State Derivative Vector
        # ---------------------------
        sdot = ptu.from_numpy(np.zeros([17]))
        sdot[0] = DynamicsDot[0]
        sdot[1] = DynamicsDot[1]
        sdot[2] = DynamicsDot[2]
        sdot[3] = DynamicsDot[3]
        sdot[4] = DynamicsDot[4]
        sdot[5] = DynamicsDot[5]
        sdot[6] = DynamicsDot[6]
        sdot[7] = DynamicsDot[7]
        sdot[8] = DynamicsDot[8]
        sdot[9] = DynamicsDot[9]
        sdot[10] = DynamicsDot[10]
        sdot[11] = DynamicsDot[11]
        sdot[12] = DynamicsDot[12]
        sdot[13] = cmd[0]/IRzz
        sdot[14] = cmd[1]/IRzz
        sdot[15] = cmd[2]/IRzz
        sdot[16] = cmd[3]/IRzz

        self.acc = sdot[7:10]

        return sdot
    
    def casadi_state_dot(self, state, cmd):
        # expects the state to be a CASADI MX symbol

        # Import Params
        # ---------------------------
        mB = self.params["mB"]
        g = self.params["g"]
        dxm = self.params["dxm"]
        dym = self.params["dym"]
        IB = self.params["IB"]
        IBxx = ptu.to_numpy(IB[0, 0])
        IByy = ptu.to_numpy(IB[1, 1])
        IBzz = ptu.to_numpy(IB[2, 2])
        Cd = self.params["Cd"]

        kTh = self.params["kTh"]
        kTo = self.params["kTo"]

        minWmotor = self.params["minWmotor"]
        maxWmotor = self.params["maxWmotor"]

        IRzz = self.params["IRzz"]
        if self.params["usePrecession"]:
            uP = 1
        else:
            uP = 0

        # Import State Vector
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

        ThrM1 = kTh * wM1 ** 2
        ThrM2 = kTh * wM2 ** 2
        ThrM3 = kTh * wM3 ** 2
        ThrM4 = kTh * wM4 ** 2
        TorM1 = kTo * wM1 ** 2
        TorM2 = kTo * wM2 ** 2
        TorM3 = kTo * wM3 ** 2
        TorM4 = kTo * wM4 ** 2

        # Wind Model (zero in expectation)
        # ---------------------------
        velW = 0
        qW1 = 0
        qW2 = 0

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
                    Cd
                    * ca.sign(velW * ca.cos(qW1) * ca.cos(qW2) - xdot)
                    * (velW * ca.cos(qW1) * ca.cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / mB,
                (
                    Cd
                    * ca.sign(velW * ca.sin(qW1) * ca.cos(qW2) - ydot)
                    * (velW * ca.sin(qW1) * ca.cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / mB,
                (
                    -Cd * ca.sign(velW * ca.sin(qW2) + zdot) * (velW * ca.sin(qW2) + zdot) ** 2
                    - (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + g * mB
                )
                / mB,
                (
                    (IByy - IBzz) * q * r
                    - uP * IRzz * (wM1 - wM2 + wM3 - wM4) * q
                    + (ThrM1 - ThrM2 - ThrM3 + ThrM4) * dym
                )
                / IBxx,  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (IBzz - IBxx) * p * r
                    + uP * IRzz * (wM1 - wM2 + wM3 - wM4) * p
                    + (ThrM1 + ThrM2 - ThrM3 - ThrM4) * dxm
                )
                / IByy,  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((IBxx - IByy) * p * q - TorM1 + TorM2 - TorM3 + TorM4) / IBzz,
                cmd[0,:]/IRzz, cmd[1,:]/IRzz, cmd[2,:]/IRzz, cmd[3,:]/IRzz
        )

        # State Derivative Vector
        # ---------------------------
        sdot = DynamicsDot

        return sdot




    def update(self, cmd, wind, t, dt): 
        prev_vel = self.vel
        prev_omega = self.omega

        # self.integrator.set_f_params(cmd, wind)
        # self.state = self.integrator.integrate(t, t+dt)
        self.state += self.state_dot(self.state, cmd, wind, t) * dt 

        self.pos = self.state[0:3]
        self.quat = self.state[3:7]
        self.vel = self.state[7:10]
        self.omega = self.state[10:13]
        self.wMotor = torch.stack(
            [self.state[13], self.state[14], self.state[15], self.state[16]]
        )

        self.vel_dot = (self.vel - prev_vel) / dt
        self.omega_dot = (self.omega - prev_omega) / dt

        self.extended_state()
        self.forces()

        return self.state

    def linmod(self, state):

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

        # Import Params
        # ---------------------------
        dxm = self.params["dxm"]
        dym = self.params["dym"]
        mB = self.params["mB"]
        IB = self.params["IB"]
        IBxx = IB[0, 0]
        IByy = IB[1, 1]
        IBzz = IB[2, 2]
        Cd = self.params["Cd"]
        IRzz = self.params["IRzz"]
        kTh = self.params["kTh"]
        kTo = self.params["kTo"]
        minWmotor = self.params["minWmotor"]
        maxWmotor = self.params["maxWmotor"]
        tau = self.params["tau"]
        kp = self.params["kp"]
        damp = self.params["damp"]

        # Motor Dynamics and Rotor forces (Second Order System: https://apmonitor.com/pdc/index.php/Main/SecondOrderSystems)
        # ---------------------------
        wMotor = torch.stack([wM1, wM2, wM3, wM4])
        wMotor = torch.clip(wMotor, minWmotor, maxWmotor)
        thrust = kTh * wMotor * wMotor
        torque = kTo * wMotor * wMotor

        ThrM1 = thrust[0]
        ThrM2 = thrust[1]
        ThrM3 = thrust[2]
        ThrM4 = thrust[3]

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
                            kTh * wM1 ** 2
                            + kTh * wM2 ** 2
                            + kTh * wM3 ** 2
                            + kTh * wM4 ** 2
                        )
                        / mB,
                        -2
                        * q3
                        * (
                            kTh * wM1 ** 2
                            + kTh * wM2 ** 2
                            + kTh * wM3 ** 2
                            + kTh * wM4 ** 2
                        )
                        / mB,
                        -2
                        * q0
                        * (
                            kTh * wM1 ** 2
                            + kTh * wM2 ** 2
                            + kTh * wM3 ** 2
                            + kTh * wM4 ** 2
                        )
                        / mB,
                        -2
                        * q1
                        * (
                            kTh * wM1 ** 2
                            + kTh * wM2 ** 2
                            + kTh * wM3 ** 2
                            + kTh * wM4 ** 2
                        )
                        / mB,
                    ]
                ),
                torch.stack(
                    [
                        2
                        * q1
                        * (
                            kTh * wM1 ** 2
                            + kTh * wM2 ** 2
                            + kTh * wM3 ** 2
                            + kTh * wM4 ** 2
                        )
                        / mB,
                        2
                        * q0
                        * (
                            kTh * wM1 ** 2
                            + kTh * wM2 ** 2
                            + kTh * wM3 ** 2
                            + kTh * wM4 ** 2
                        )
                        / mB,
                        -2
                        * q3
                        * (
                            kTh * wM1 ** 2
                            + kTh * wM2 ** 2
                            + kTh * wM3 ** 2
                            + kTh * wM4 ** 2
                        )
                        / mB,
                        -2
                        * q2
                        * (
                            kTh * wM1 ** 2
                            + kTh * wM2 ** 2
                            + kTh * wM3 ** 2
                            + kTh * wM4 ** 2
                        )
                        / mB,
                    ]
                ),
                torch.stack(
                    [
                        -2
                        * q0
                        * (
                            kTh * wM1 ** 2
                            + kTh * wM2 ** 2
                            + kTh * wM3 ** 2
                            + kTh * wM4 ** 2
                        )
                        / mB,
                        2
                        * q1
                        * (
                            kTh * wM1 ** 2
                            + kTh * wM2 ** 2
                            + kTh * wM3 ** 2
                            + kTh * wM4 ** 2
                        )
                        / mB,
                        2
                        * q2
                        * (
                            kTh * wM1 ** 2
                            + kTh * wM2 ** 2
                            + kTh * wM3 ** 2
                            + kTh * wM4 ** 2
                        )
                        / mB,
                        -2
                        * q3
                        * (
                            kTh * wM1 ** 2
                            + kTh * wM2 ** 2
                            + kTh * wM3 ** 2
                            + kTh * wM4 ** 2
                        )
                        / mB,
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
                            Cd
                            * (-2 * velW * cos(qW1) * cos(qW2) + 2 * xdot)
                            * sign(velW * cos(qW1) * cos(qW2) - xdot)
                        )
                        / mB,
                        _,
                        _,
                    ]
                ),
                torch.stack(
                    [
                        _,
                        (
                            Cd
                            * (-2 * velW * sin(qW1) * cos(qW2) + 2 * ydot)
                            * sign(velW * sin(qW1) * cos(qW2) - ydot)
                        )
                        / mB,
                        _,
                    ]
                ),
                torch.stack(
                    [
                        _,
                        _,
                        (
                            -Cd
                            * (2 * velW * sin(qW2) + 2 * zdot)
                            * sign(velW * sin(qW2) + zdot)
                        )
                        / mB,
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
                        (-IRzz * (wM1 - wM2 + wM3 - wM4) + r * (IByy - IBzz)) / IBxx,
                        q * (IByy - IBzz) / IBxx,
                    ]
                ),
                torch.stack(
                    [
                        (IRzz * (wM1 - wM2 + wM3 - wM4) + r * (-IBxx + IBzz)) / IByy,
                        _,
                        p * (-IBxx + IBzz) / IByy,
                    ]
                ),
                torch.stack([q * (IBxx - IByy) / IBzz, p * (IBxx - IByy) / IBzz, _]),
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
                        -2 * kTh * wM1 * (2 * q0 * q2 + 2 * q1 * q3) / mB,
                        -2 * kTh * wM2 * (2 * q0 * q2 + 2 * q1 * q3) / mB,
                        -2 * kTh * wM3 * (2 * q0 * q2 + 2 * q1 * q3) / mB,
                        -2 * kTh * wM4 * (2 * q0 * q2 + 2 * q1 * q3) / mB,
                    ]
                ),
                torch.stack(
                    [
                        2 * kTh * wM1 * (2 * q0 * q1 - 2 * q2 * q3) / mB,
                        2 * kTh * wM2 * (2 * q0 * q1 - 2 * q2 * q3) / mB,
                        2 * kTh * wM3 * (2 * q0 * q1 - 2 * q2 * q3) / mB,
                        2 * kTh * wM4 * (2 * q0 * q1 - 2 * q2 * q3) / mB,
                    ]
                ),
                torch.stack(
                    [
                        -2 * kTh * wM1 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / mB,
                        -2 * kTh * wM2 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / mB,
                        -2 * kTh * wM3 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / mB,
                        -2 * kTh * wM4 * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) / mB,
                    ]
                ),
                torch.stack(
                    [
                        (-IRzz * q + 2 * dym * kTh * wM1) / IBxx,
                        (IRzz * q - 2 * dym * kTh * wM2) / IBxx,
                        (-IRzz * q - 2 * dym * kTh * wM3) / IBxx,
                        (IRzz * q + 2 * dym * kTh * wM4) / IBxx,
                    ]
                ),
                torch.stack(
                    [
                        (IRzz * p + 2 * dxm * kTh * wM1) / IByy,
                        (-IRzz * p + 2 * dxm * kTh * wM2) / IByy,
                        (IRzz * p - 2 * dxm * kTh * wM3) / IByy,
                        (-IRzz * p - 2 * dxm * kTh * wM4) / IByy,
                    ]
                ),
                torch.stack(
                    [
                        -2 * kTo * wM1 / IBzz,
                        2 * kTo * wM2 / IBzz,
                        -2 * kTo * wM3 / IBzz,
                        2 * kTo * wM4 / IBzz,
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
                torch.stack([_ + 1 / IRzz, _, _, _]),
                torch.stack([_, _ + 1 / IRzz, _, _]),
                torch.stack([_, _, _ + 1 / IRzz, _]),
                torch.stack([_, _, _, _ + 1 / IRzz]),
            ]
        )

        return A, B
