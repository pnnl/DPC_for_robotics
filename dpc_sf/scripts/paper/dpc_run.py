"""
This script is designed to run all DPC scenarios
- waypoint point to point navigation around obstacles (wp_p2p)
- straight line trajectory reference tracking (wp_traj)
- sinusoidal trajectory reference tracking (fig8)
"""

import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

from dpc_sf.control.dpc.run_p2p import run_dpc_p2p
from dpc_sf.control.dpc.run_p2p_traj import run_wp_traj
from dpc_sf.control.dpc.run_fig8 import run_dpc_fig8

run_dpc_p2p(
    Ts=0.001,
    nstep=5000,
    save_path="data/policy/DPC_p2p/policy_experimental_tt.pth"
)

run_wp_traj(
    Ts=0.001,
    nstep=15000, # needs to be 20k, but only 15k works on gpu
    average_vel=0.5,
    save_path="data/policy/DPC_traj/policy.pth"
)

run_dpc_fig8(
    Ts=0.001,
    nstep=10000,
    average_vel=1.0,
    save_path="data/policy/DPC_fig8/policy.pth"
)

