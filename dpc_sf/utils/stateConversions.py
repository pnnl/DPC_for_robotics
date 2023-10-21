# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

"""First pass on pytorch conversion complete"""

import torch
from torch import sin, cos
import dpc_sf.utils
import numpy as np

def sDes2state(sDes):
    pos = sDes[0:3]
    vel = sDes[3:6]
    acc = sDes[6:9]
    thr = sDes[9:12]
    eul = sDes[12:15]
    pqr = sDes[15:18]
    des_yaw_rate = sDes[18]

    # we don't care about attitude
    quat = dpc_sf.utils.YPRToQuat(eul[0], eul[1], eul[2])

    # we don't care about rotational rates of rotors
    omegas = np.array([522.9847412109375]*4)

    state = np.hstack([pos, quat, acc, pqr, omegas])

    return state

def phiThetaPsiDotToPQR(phi, theta, psi, phidot, thetadot, psidot: torch.Tensor) -> torch.Tensor:
    
    p = -sin(theta)*psidot + phidot
    
    q = sin(phi)*cos(theta)*psidot + cos(phi)*thetadot
    
    r = -sin(phi)*thetadot + cos(phi)*cos(theta)*psidot
    
    return torch.cat([p, q, r])


def xyzDotToUVW_euler(phi, theta, psi, xdot, ydot, zdot: torch.Tensor) -> torch.Tensor:
    u = xdot*cos(psi)*cos(theta) + ydot*sin(psi)*cos(theta) - zdot*sin(theta)
    
    v = (sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi))*ydot + (sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi))*xdot + zdot*sin(phi)*cos(theta)
    
    w = (sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi))*xdot + (-sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi))*ydot + zdot*cos(phi)*cos(theta)
    
    return torch.cat([u, v, w])


def xyzDotToUVW_Flat_euler(phi, theta, psi, xdot, ydot, zdot: torch.Tensor) -> torch.Tensor:
    uFlat = xdot * cos(psi) + ydot * sin(psi)

    vFlat = -xdot * sin(psi) + ydot * cos(psi)

    wFlat = zdot

    return torch.cat([uFlat, vFlat, wFlat])

def xyzDotToUVW_Flat_quat(q, xdot, ydot, zdot: torch.Tensor) -> torch.Tensor:
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    uFlat = 2*(q0*q3 - q1*q2)*ydot + (q0**2 - q1**2 + q2**2 - q3**2)*xdot

    vFlat = -2*(q0*q3 + q1*q2)*xdot + (q0**2 + q1**2 - q2**2 - q3**2)*ydot

    wFlat = zdot

    return torch.cat([uFlat, vFlat, wFlat])
