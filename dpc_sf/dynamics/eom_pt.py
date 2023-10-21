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

import copy
import torch
from torch import sin, cos, tan, pi, sign
import numpy as np
from dpc_sf.control.trajectory.trajectory import waypoint_reference

from dpc_sf.utils import pytorch_utils as ptu
from dpc_sf.utils.animation import Animator

from dpc_sf.dynamics.params import params

def state_dot_np(state, cmd, params):
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

def state_dot_nm(state, cmd, params, include_actuators=True):

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

    if include_actuators:

        wM1 =   state[...,13]
        wM2 =   state[...,14]
        wM3 =   state[...,15]
        wM4 =   state[...,16]

    else:

        wM1 =   cmd[...,0]
        wM2 =   cmd[...,1]
        wM3 =   cmd[...,2]
        wM4 =   cmd[...,3]

    wMotor = torch.stack([wM1, wM2, wM3, wM4])
    wMotor = torch.clip(wMotor, params["minWmotor"], params["maxWmotor"])
    thrust = params["kTh"] * wMotor ** 2
    torque = params["kTo"] * wMotor ** 2

    # Wind Model
    # ---------------------------
    velW, qW1, qW2 = [ptu.from_numpy(np.array(0))]*3

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
                * sign(velW * cos(qW1) * cos(qW2) - xdot)
                * (velW * cos(qW1) * cos(qW2) - xdot) ** 2
                - 2 * (q0 * q2 + q1 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
            )
            / params["mB"],
            (
                params["Cd"]
                * sign(velW * sin(qW1) * cos(qW2) - ydot)
                * (velW * sin(qW1) * cos(qW2) - ydot) ** 2
                + 2 * (q0 * q1 - q2 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
            )
            / params["mB"],
            (
                -params["Cd"] * sign(velW * sin(qW2) + zdot) * (velW * sin(qW2) + zdot) ** 2
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

    if include_actuators:
        ActuatorsDot = cmd/params["IRzz"]
        return torch.hstack([DynamicsDot.T, ActuatorsDot])
    else:
        return DynamicsDot.T


def state_dot_pt(state, cmd, params=params, include_actuators=True):

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

    if include_actuators:
        wM1 =   state[:,13]
        wM2 =   state[:,14]
        wM3 =   state[:,15]
        wM4 =   state[:,16]
    else:
        wM1 = cmd[:,0]
        wM2 = cmd[:,1]
        wM3 = cmd[:,2]
        wM4 = cmd[:,3]

    wMotor = torch.stack([wM1, wM2, wM3, wM4])
    wMotor = torch.clip(wMotor, params["minWmotor"], params["maxWmotor"])
    thrust = params["kTh"] * wMotor ** 2
    torque = params["kTo"] * wMotor ** 2

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
                params["Cd"]
                * sign(velW * cos(qW1) * cos(qW2) - xdot)
                * (velW * cos(qW1) * cos(qW2) - xdot) ** 2
                - 2 * (q0 * q2 + q1 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
            )
            / params["mB"],
            (
                params["Cd"]
                * sign(velW * sin(qW1) * cos(qW2) - ydot)
                * (velW * sin(qW1) * cos(qW2) - ydot) ** 2
                + 2 * (q0 * q1 - q2 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
            )
            / params["mB"],
            (
                -params["Cd"] * sign(velW * sin(qW2) + zdot) * (velW * sin(qW2) + zdot) ** 2
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

    if include_actuators:
        ActuatorsDot = cmd/params["IRzz"]

        # State Derivative Vector
        # ---------------------------
        if state.shape[0] == 1:
            return torch.hstack([DynamicsDot.squeeze(), ActuatorsDot.squeeze()])
        else:
            return torch.hstack([DynamicsDot.T, ActuatorsDot])
    else:
        if state.shape[0] == 1:
            return DynamicsDot.squeeze()
        else:
            return DynamicsDot.T

# This quadcopter class contains no state, only simulation parameters and function
class QuadcopterPT:
    def __init__(
            self, 
            state: np.ndarray = params["default_init_state_np"], # initial condition
            reference = waypoint_reference(type='wp_p2p', average_vel=1.6),
            params=params,
            Ts: float = 0.1,
            Ti: float = 0.0,
            Tf: float = 4.0,
            integrator='euler', # 'euler', 'rk4'
            include_actuators=True
    ):

        # Whether or not to expect neuromancer variables rather than torch tensors for state and command
        self.integrator = integrator
        self.reference = reference
        self.params = params
        if include_actuators is True:
            self.state = state
        else:
            self.state = state[0:13]
        self.Ts = Ts
        self.Ti = Ti
        self.Tf = Tf
        self.t = Ti
        self.cmd = np.array([0,0,0,0])
        self.include_actuators = include_actuators

        # for animation:
        self.state_history = []
        self.time_history = []
        self.reference_history = []
        self.input_history = []


    def step(
            self, 
            cmd: np.ndarray,
        ):

        assert isinstance(cmd, np.ndarray)

        self.cmd = cmd
        self.save_state()

        cmd = ptu.from_numpy(cmd)

        state_pt = ptu.from_numpy(self.get_state())

        if self.integrator == 'euler':
            state_pt += self.state_dot(state_pt, cmd) * self.Ts
        
        elif self.integrator == 'rk4':
            k1 = self.state_dot(state_pt, cmd)
            k2 = self.state_dot(state_pt + self.Ts/2 * k1, cmd)
            k3 = self.state_dot(state_pt + self.Ts/2 * k2, cmd)
            k4 = self.state_dot(state_pt + self.Ts * k3, cmd)
            state_pt += self.Ts/6 * (k1 + 2 * k2 + 2 * k3 + k4)

        self.state = ptu.to_numpy(state_pt)

        self.t += self.Ts

    def reset(
            self,
            state
        ):
        self.set_state(copy.deepcopy(state))
        self.t = self.Ti
        self.state_history = []
        self.time_history = []
        self.reference_history = []
        self.input_history = []

    def get_state(
            self
        ):
        if isinstance(self.state, torch.Tensor):
            return ptu.to_numpy(self.state)
        else:
            return self.state
    
    def set_state(
            self,
            state: np.ndarray
        ):
        self.state = ptu.from_numpy(state)

    def save_state(self):
        # consistent saving scheme for mj and eom
        self.state_history.append(copy.deepcopy(self.state)) # deepcopy required, tensor stored by reference
        self.time_history.append(self.t) # no copy required as it is a float, not stored by reference
        self.reference_history.append(np.copy(self.reference(self.t))) # np.copy great as reference is a np.array
        self.input_history.append(np.copy(self.cmd)) # np.co py great as reference is a np.array

    def animate(self, state_prediction=None, render_interval=1):

        if self.reference.type == 'wp_p2p':
            drawCylinder = True
        else:
            drawCylinder = False

        if state_prediction is not None:
            animator = Animator(
                states=np.vstack(self.state_history)[::render_interval,:], 
                times=np.array(self.time_history)[::render_interval], 
                reference_history=np.vstack(self.reference_history)[::render_interval,:], 
                reference=self.reference, 
                reference_type=self.reference.type, 
                drawCylinder=drawCylinder,
                state_prediction=state_prediction[::render_interval,...]
            )
        else:
            animator = Animator(
                states=np.vstack(self.state_history)[::render_interval,:], 
                times=np.array(self.time_history)[::render_interval], 
                reference_history=np.vstack(self.reference_history)[::render_interval,:], 
                reference=self.reference, 
                reference_type=self.reference.type, 
                drawCylinder=drawCylinder,
                state_prediction=state_prediction
            )

        animator.animate() # does not contain plt.show()

    def state_dot(self, state: torch.Tensor, cmd: torch.Tensor):
        return state_dot_pt(state, cmd, self.params, self.include_actuators)

    
if __name__ == "__main__":
    # running a main() inside the __name__ __main__ keeps vscode outline variables contained
    def main():
        # quick check to validate it is equivalent to old quad
        from dpc_sf.control.mpc.mpc import MPC_Point_Ref
        from dpc_sf.control.trajectory.trajectory import waypoint_reference
        from dpc_sf.dynamics.eom_ca import QuadcopterCA

        quadCA = QuadcopterCA(params=params)

        Ts = 0.1
        Ti = 0.0
        Tf = 4.0
        integrator = 'euler'

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
        ]))

        reference = waypoint_reference(type='wp_p2p', average_vel=1.6)

        quad = QuadcopterPT(
            state=state,
            reference=reference,
            params=params,
            Ts=Ts,
            Ti=Ti,
            Tf=Tf,
            integrator=integrator
        )

        ctrl = MPC_Point_Ref(
            N=30,
            dt=Ts,
            interaction_interval=1,
            n=17,
            m=4,
            dynamics=quadCA.state_dot,
            state_ub=quadCA.params['ca_state_ub'],
            state_lb=quadCA.params['ca_state_lb'],
            return_type='numpy',
            obstacle=True,
            integrator_type=integrator
        )

        ctrl_pred_x = []
        while quad.t < quad.Tf:

            cmd = ctrl(quad.state, reference(quad.t))
            quad.step(cmd)

            ctrl_predictions = ctrl.get_predictions() 
            ctrl_pred_x.append(ctrl_predictions[0])


        ctrl_pred_x = np.stack(ctrl_pred_x)
        quad.animate(state_prediction=ctrl_pred_x)
        

    main()
