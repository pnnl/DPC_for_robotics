import matplotlib.pyplot as plt
import numpy as np
from utils.quad import plot_mujoco_trajectories_wp_traj, plot_mujoco_trajectories_wp_p2p, draw_cylinder
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

# plot_mujoco_trajectories_wp_p2p(nmpc_nav, 'data/nav/nmpc_nav.svg')
# plot_mujoco_trajectories_wp_p2p(vtnmpc_nav, 'data/nav/vtnmpc_nav.svg')
# plot_mujoco_trajectories_wp_p2p(dpc_nav, 'data/nav/dpc_nav.svg')
# plot_mujoco_trajectories_wp_p2p(dpc_sf_nav, 'data/nav/dpc_sf_nav.svg')

def plot_2d_trajectories(dpc, dpc_sf, vtnmpc, nmpc, filename):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    sets = [dpc, dpc_sf, vtnmpc, nmpc]  # List of all trajectory sets
    set_names = ['DPC', 'DPC + PSF', 'VTNMPC', 'NMPC']
    for i, ax in enumerate(axs.flatten()):
        current_set = sets[i]
        if current_set is not None:
            for output in current_set:
                # Assuming the tensor is already in a numpy-compatible format, if not, convert it accordingly
                x = output['X'][0, 1:, 0].numpy()  # Extracting the x positions
                y = output['X'][0, 1:, 1].numpy()  # Extracting the y positions
                ax.plot(x, y)
            circle = plt.Circle((1, 1), 0.5, color='r', fill=False)
            ax.add_patch(circle)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(set_names[i])
            ax.set_aspect('equal', 'box')
            ax.grid(True)
            # ax.legend()

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    # plt.show()
    plt.savefig(filename)
    plt.close(fig)

    print('Plot saved to', filename)

plot_2d_trajectories(dpc_nav, dpc_sf_nav, vtnmpc_nav, nmpc_nav, 'data/nav/nav.svg')

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

nmpc_adv_nav = load_nav('data/adv_nav/nmpc_adv_nav.npz', flip_z=True)
vtnmpc_adv_nav = load_nav('data/adv_nav/vtnmpc_adv_nav.npz', flip_z=True)
dpc_adv_nav = load_nav('data/adv_nav/dpc_adv_nav.npz', flip_z=False)
dpc_sf_adv_nav = load_nav('data/adv_nav/dpc_sf_adv_nav.npz', flip_z=False)

# plot_mujoco_trajectories_wp_p2p(nmpc_adv_nav, 'data/adv_nav/nmpc_adv_nav.svg')
# plot_mujoco_trajectories_wp_p2p(vtnmpc_adv_nav, 'data/adv_nav/vtnmpc_adv_nav.svg')
# plot_mujoco_trajectories_wp_p2p(dpc_adv_nav, 'data/adv_nav/dpc_adv_nav.svg')
# plot_mujoco_trajectories_wp_p2p(dpc_sf_adv_nav, 'data/adv_nav/dpc_sf_adv_nav.svg')

plot_2d_trajectories(dpc_adv_nav, dpc_sf_adv_nav, vtnmpc_adv_nav, nmpc_adv_nav, 'data/adv_nav/adv_nav.svg')

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

def plot_3d_trajectories(dpc, dpc_sf, vtnmpc, nmpc, filename):
    fig = plt.figure(figsize=(10, 10))
    sets = [dpc, dpc_sf, vtnmpc, nmpc]  # List of all trajectory sets
    set_names = ['DPC', 'DPC + SF', 'VTNMPC', 'NMPC']

    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        current_set = sets[i]
        if current_set is not None:
            # Assuming the tensor is already in a numpy-compatible format, if not, convert it accordingly
            x = current_set['X'][0, 1:, 0].numpy()  # Extracting the x positions
            y = -current_set['X'][0, 1:, 1].numpy()  # Extracting the y positions
            z = -current_set['X'][0, 1:, 2].numpy()  # Extracting the z positions, assuming 3D trajectories
            xr = current_set['R'][0, 1:, 0].numpy()  # Extracting the x positions
            yr = -current_set['R'][0, 1:, 1].numpy()  # Extracting the y positions
            zr = -current_set['R'][0, 1:, 2].numpy()  #
            ax.plot(x, y, z, 'b', label='Trajectory')
            ax.plot(xr, yr, zr, 'g--', label='Reference')
            # Plot start position in green
            ax.scatter(x[0], y[0], z[0], color='green', s=50, label='Start')
            # Plot end position in red
            ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, label='End')

            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Z Position')
            ax.set_title(set_names[i])
            if i == 0: ax.legend()

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(filename)
    plt.close(fig)

    print('Plot saved to', filename)

# plot_mujoco_trajectories_wp_traj([nmpc_traj], 'data/traj/nmpc_traj.svg')
# plot_mujoco_trajectories_wp_traj([vtnmpc_traj], 'data/traj/vtnmpc_traj.svg')
# plot_mujoco_trajectories_wp_traj([dpc_sf_traj], 'data/traj/dpc_sf_traj.svg')
# plot_mujoco_trajectories_wp_traj([dpc_traj], 'data/traj/dpc_traj.svg')

plot_3d_trajectories(dpc_traj, dpc_sf_traj, vtnmpc_traj, nmpc_traj, 'data/traj/traj.svg')

print('fin')
