"""
This script is designed to perform training of all DPC scenarios and save policies
- waypoint point to point navigation around obstacles (wp_p2p)
- straight line trajectory reference tracking (wp_traj)
- sinusoidal trajectory reference tracking (fig8)
"""

import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

import dpc_sf.utils.pytorch_utils as ptu
from dpc_sf.control.dpc.train_p2p_minimal import train_wp_p2p
from dpc_sf.control.dpc.train_p2p_traj_minimal import train_wp_traj
from dpc_sf.control.dpc.train_fig8_minimal import train_fig8

ptu.init_gpu(use_gpu=True)

train_wp_p2p(
    epochs=30,
    iterations=1,
    save_path="data/policy/DPC_p2p/",
    media_path="data/media/dpc/images/"
)

train_wp_traj(
    epochs=10,
    iterations=1,
    save_path="data/policy/DPC_traj/",
    media_path="data/media/dpc/images/"
)

train_fig8(
    epochs=10,
    iterations=1,
    save_path="data/policy/DPC_fig8/",
    media_path="data/media/dpc/images/"
)

