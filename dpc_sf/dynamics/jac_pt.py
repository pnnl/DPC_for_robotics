from dpc_sf.dynamics.params import params

import dpc_sf.utils.pytorch_utils as ptu
import torch
from torch import sin, cos, sign
import numpy as np

class QuadcopterJac:
    def __init__(self, params=params) -> None:
        self.params = params

    def __call__(self, state):
        return self.linmod(state)

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