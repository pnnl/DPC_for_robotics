"""
The purpose of this code is to have a pure
"""

import copy
import casadi as ca
import torch
from torch import sin, cos, tan, pi, sign
import numpy as np

from dpc_sf.utils import pytorch_utils as ptu
from dpc_sf.dynamics.params import params

# This quadcopter class contains no state, only simulation parameters and function
class QuadcopterNM(torch.nn.Module):
    def __init__(self, params=params) -> None:
        super().__init__()
        self.params = params
        

    def forward(self, state: torch.Tensor, cmd: torch.Tensor):

        def ensure_2d(tensor):
            if len(tensor.shape) == 1:
                tensor = torch.unsqueeze(tensor, 0)
            return tensor
        
        state = ensure_2d(state)
        cmd = ensure_2d(cmd)

        # Unpack state tensor for readability
        # ---------------------------

        # this unbind method works on raw tensors, but not NM variables
        _, _, _, q0, q1, q2, q3, xdot, ydot, zdot, p, q, r, wM1, wM2, wM3, wM4 = torch.unbind(state, dim=1)
        # try the state.unpack function that comes with the nm

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
