"""
This script is to test the orientation and thrust control
"""

import numpy as np

from dpc_sf.control.dpc.dpc_learn4 import CtrlThrustPitchRoll
from dpc_sf.dynamics.eom_pt import QuadcopterPT
from dpc_sf.utils.random_state import random_state
from dpc_sf.utils import pytorch_utils as ptu

Ts=0.1

quad = QuadcopterPT(Ts=Ts)

# choose no normalization with the means and variances - TODO: TEST THIS
tpr_norm_2_u = CtrlThrustPitchRoll(
    Ts=Ts,
    mlp_umax=1,
    mlp_umin=-1,
    normalize=True,
    mean=np.array([
        0,0,0,
        0,0,0,0,
        0,0,0,
        0,0,0,
        0,0,0,0
    ]),
    var=np.array([
        1,1,1,
        1,1,1,1,
        1,1,1,
        1,1,1,
        1,1,1,1
    ])
)

# lets get a random state

quad.set_state(state=random_state())

tpr_des = ptu.from_numpy(np.array([[
    quad.params["kTh"] * quad.params["w_hover"] ** 2,
    0,
    0
]]))

for i in range(10):
    x = ptu.from_numpy(quad.get_state()).unsqueeze(0)
    u = ptu.to_numpy(tpr_norm_2_u(x_norm=x, tpr_norm=tpr_des))
    quad.step(u)

print('fin')