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
import casadi as ca
import torch
from torch import sin, cos, tan, pi, sign
import numpy as np
from dpc_sf.utils import pytorch_utils as ptu
from dpc_sf.dynamics.params import params
from neuromancer.modules.blocks import MLP
from dpc_sf.dynamics.eom_pt import state_dot_pt

class TestSysID(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()


# This quadcopter class contains no state, only simulation parameters and function
class QuadcopterSysID(torch.nn.Module):
    def __init__(self, params=params) -> None:
        super().__init__()
        self.params = params
        self.in_features, self.out_features = 21, 17

        # recieves the velocity and tries to find residual
        self.pos_block = MLP(
            insize=3,
            outsize=3,
            hsizes=[16,16]
        )
        # recieves the ang vel and tries to find residual of them
        self.quat_block = MLP(
            insize=3,
            outsize=4,
            hsizes=[16,16]
        )
        # recieves the rotor omegas and tries correct body rates
        self.av_block = MLP(
            insize=4,
            outsize=3,
            hsizes=[16, 16]
        )

        self.full_block = MLP(
            insize=21,
            outsize=17,
            hsizes=[32,32]
        )

        self.beta = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=True)

    def forward(self, state_cmd: torch.tensor):

        state = state_cmd[:,:17]
        cmd = state_cmd[:,17:]

        # the baseline known dynamics
        state_dot = state_dot_pt(state, cmd, self.params)#  * 1.5 * self.beta

        def ensure_2d(tensor):
            if len(tensor.shape) == 1:
                tensor = torch.unsqueeze(tensor, 0)
            return tensor
        
        state_dot = ensure_2d(state_dot)

        # correct the updates
        # pos_correction = self.pos_block(state[:, 7:10])
        # quat_correction = self.quat_block(state[:, 10:13])
        # av_correction = self.av_block(state[:,13:17])

        # state_dot[:,0:3] += pos_correction
        # state_dot[:,3:7] += quat_correction
        # state_dot[:,10:13] += av_correction

        state_dot += self.full_block(state_cmd)
        # state_dot += self.full_block(state_cmd)

        return state_dot
