# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import torch
import numpy as np
import dpc_sf.utils.pytorch_utils as ptu
# from dpc_sf.quad_dynamics.params import params as quad_params
import casadi as ca

"""First pass on pytorch conversion complete"""

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

def mixerFM_np(params, thr, moment):
    t = np.array([thr, moment[0], moment[1], moment[2]])
    w_cmd = np.sqrt(np.clip(np.dot(params["mixerFMinv"], t), params["minWmotor"]**2, params["maxWmotor"]**2))
    return w_cmd

def mixerFM(params: dict, thr: torch.Tensor, moment: torch.Tensor) -> torch.Tensor:
    t = torch.stack([thr, moment[0], moment[1], moment[2]])
    w_cmd = torch.sqrt(torch.clip(params["mixerFMinv"] @ t, params["minWmotor"]**2, params["maxWmotor"]**2))
    return w_cmd

def mixerFM_batched(params: dict, thr: torch.Tensor, moment: torch.Tensor) -> torch.Tensor:
    t = torch.stack([thr, moment[:,0], moment[:,1], moment[:,2]], dim=1)

    # Perform batched matrix multiplication
    w_cmd_sq = torch.bmm(params["mixerFMinv"].unsqueeze(0).expand(t.size(0), -1, -1), t.unsqueeze(-1)).squeeze(-1)

    # Clip and take square root
    w_cmd = torch.sqrt(torch.clamp(w_cmd_sq, min=params["minWmotor"]**2, max=params["maxWmotor"]**2))
    
    # w_cmd = torch.sqrt(torch.clip(params["mixerFMinv"] @ t, params["minWmotor"]**2, params["maxWmotor"]**2))
    return w_cmd

def mixerFM_batched_ca(params: dict, thr, moment):
    t = ca.vertcat(thr, moment[:,0], moment[:,1], moment[:,2])

    mixer = ca.DM(ptu.to_numpy(params["mixerFMinv"]))
    # Perform matrix multiplication
    w_cmd_sq = ca.mtimes(mixer, t)

    # Clamp and take square root
    w_cmd_sq_clamped = ca.fmax(ca.fmin(w_cmd_sq, params["maxWmotor"]**2), params["minWmotor"]**2)
    w_cmd = ca.sqrt(w_cmd_sq_clamped)
    
    return w_cmd

## Under here is the conventional type of mixer

# def mixer(throttle, pCmd, qCmd, rCmd, quad):
#     maxCmd = quad.params["maxCmd"]
#     minCmd = quad.params["minCmd"]

#     cmd = np.zeros([4, 1])
#     cmd[0] = throttle + pCmd + qCmd - rCmd
#     cmd[1] = throttle - pCmd + qCmd + rCmd
#     cmd[2] = throttle - pCmd - qCmd - rCmd
#     cmd[3] = throttle + pCmd - qCmd + rCmd
    
#     cmd[0] = min(max(cmd[0], minCmd), maxCmd)
#     cmd[1] = min(max(cmd[1], minCmd), maxCmd)
#     cmd[2] = min(max(cmd[2], minCmd), maxCmd)
#     cmd[3] = min(max(cmd[3], minCmd), maxCmd)
    
#     # Add Exponential to command
#     # ---------------------------
#     cmd = expoCmd(quad.params, cmd)

#     return cmd

# def expoCmd(params, cmd):
#     if params["ifexpo"]:
#         cmd = np.sqrt(cmd)*10
    
#     return cmd

# def expoCmdInv(params, cmd):
#     if params["ifexpo"]:
#         cmd = (cmd/10)**2
    
#     return cmd
