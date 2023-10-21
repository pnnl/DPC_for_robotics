"""Validate jacobian against the functorch result using the pytorch graph"""
from dpc_sf.dynamics.eom_pt import QuadcopterPT
from dpc_sf.control.trajectory.trajectory import waypoint_reference
from dpc_sf.dynamics.params import params
import torch
import numpy as np
import dpc_sf.utils.pytorch_utils as ptu

def validate_jacobian():
    
    print('validating jacobian...')

    cmd = torch.zeros(4)

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

    reference = waypoint_reference('wp_p2p', average_vel=1.0)

    quad = QuadcopterPT(
        state=ptu.to_numpy(state),
        reference=reference,
        params=params,
        Ts=0.1,
        Ti=0,
        Tf=4,
        integrator='euler',
    )

    def xdot_wrapper(state, cmd):
        return quad.state_dot(state, cmd)

    A, B = torch.func.jacrev(xdot_wrapper, argnums=(0,1))(state, cmd)

    A_test, B_test = quad.linmod(state)

    assert (A-A_test).abs().max() <= 1e-05
    assert (B-B_test).abs().max() <= 1e-08
    print('passed')