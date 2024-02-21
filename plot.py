import matplotlib.pyplot as plt
import numpy as np
from utils.quad import calculate_mpc_cost, plot_mujoco_trajectories_wp_traj, plot_mujoco_trajectories_wp_p2p
import os
import utils.pytorch as ptu

# nav task

def load_nav(file, flip_z=True):
    data = np.load(file)
    output = []
    for i in range(5):
        x = ptu.from_numpy(data[data.files[0 + i*2]][None,...])
        if flip_z is True:
            x[:,:,2] *= -1
        output.append({
            'X': x,
            'U': ptu.from_numpy(data[data.files[1 + i*2]][None,...]),
            })
    return output

nmpc_nav = load_nav('data/nav/nmpc_nav.npz', flip_z=True)
vtnmpc_nav = load_nav('data/nav/vtnmpc_nav.npz', flip_z=False)
dpc_nav = load_nav('data/nav/dpc_nav.npz', flip_z=False)
dpc_sf_nav = load_nav('data/nav/dpc_sf_nav.npz', flip_z=False)

plot_mujoco_trajectories_wp_p2p(nmpc_nav, 'data/nav/nmpc_nav.svg')
plot_mujoco_trajectories_wp_p2p(vtnmpc_nav, 'data/nav/vtnmpc_nav.svg')
plot_mujoco_trajectories_wp_p2p(dpc_nav, 'data/nav/dpc_nav.svg')
plot_mujoco_trajectories_wp_p2p(dpc_sf_nav, 'data/nav/dpc_sf_nav.svg')

# adv nav task

def load_nav(file, flip_z=True):
    data = np.load(file)
    output = []
    for i in range(5):
        x = ptu.from_numpy(data[data.files[0 + i*2]][None,...])
        if flip_z is True:
            x[:,:,2] *= -1
        output.append({
            'X': x,
            'U': ptu.from_numpy(data[data.files[1 + i*2]][None,...]),
            })
    return output

nmpc_nav = load_nav('data/adv_nav/nmpc_adv_nav.npz', flip_z=True)
vtnmpc_nav = load_nav('data/adv_nav/vtnmpc_adv_nav.npz', flip_z=True)
dpc_nav = load_nav('data/adv_nav/dpc_adv_nav.npz', flip_z=False)
dpc_sf_nav = load_nav('data/adv_nav/dpc_sf_adv_nav.npz', flip_z=False)

plot_mujoco_trajectories_wp_p2p(nmpc_nav, 'data/adv_nav/nmpc_adv_nav.svg')
plot_mujoco_trajectories_wp_p2p(vtnmpc_nav, 'data/adv_nav/vtnmpc_adv_nav.svg')
plot_mujoco_trajectories_wp_p2p(dpc_nav, 'data/adv_nav/dpc_adv_nav.svg')
plot_mujoco_trajectories_wp_p2p(dpc_sf_nav, 'data/adv_nav/dpc_sf_adv_nav.svg')

# traj task

def load_traj(file):
    data = np.load(file)
    return {
           'X': ptu.from_numpy(data[data.files[0]][None,...]),
           'U': ptu.from_numpy(data[data.files[1]][None,...]),
           'R': ptu.from_numpy(data[data.files[2]][None,...]), 
           }

nmpc_traj = load_traj('data/traj/nmpc_traj.npz')
vtnmpc_traj = load_traj('data/traj/vtnmpc_traj.npz')
dpc_sf_traj = load_traj('data/traj/dpc_sf_traj.npz')
dpc_traj = load_traj('data/traj/dpc_traj.npz')

plot_mujoco_trajectories_wp_traj([nmpc_traj], 'data/traj/nmpc_traj.svg')
plot_mujoco_trajectories_wp_traj([vtnmpc_traj], 'data/traj/vtnmpc_traj.svg')
plot_mujoco_trajectories_wp_traj([dpc_sf_traj], 'data/traj/dpc_sf_traj.svg')
plot_mujoco_trajectories_wp_traj([dpc_traj], 'data/traj/dpc_traj.svg')


