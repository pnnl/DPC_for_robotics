import matplotlib.pyplot as plt
import numpy as np
import os
import utils.pytorch as ptu
from utils.quad import Animator
from dynamics import mujoco_quad, get_quad_params
import imageio

###### Paper #######

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

# plot_2d_trajectories(dpc_nav, dpc_sf_nav, vtnmpc_nav, nmpc_nav, 'data/nav/nav.svg')

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

# plot_2d_trajectories(dpc_adv_nav, dpc_sf_adv_nav, vtnmpc_adv_nav, nmpc_adv_nav, 'data/adv_nav/adv_nav.svg')

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

# plot_3d_trajectories(dpc_traj, dpc_sf_traj, vtnmpc_traj, nmpc_traj, 'data/traj/traj.svg')

def plot_mujoco_trajectories(traj, filename):
    quad_params = get_quad_params()
    state = quad_params["default_init_state_np"]
    Ti, Tf, Ts = 0.0, 20.0, 0.001
    mj_quad = mujoco_quad(state=state, quad_params=quad_params, Ti=Ti, Tf=Tf, Ts=Ts, integrator='euler', render='mujoco')
    mj_quad.images = []
    traj = ptu.to_numpy(traj['X'][0])
    for i, x in enumerate(traj):
        if i % 400 == 0:
            mj_quad.set_state(x)
    video = np.stack(mj_quad.images)
    # We take the maximum value across the 0th axis (across all frames)
    timelapse_image = np.max(video, axis=0)
    plt.imsave(filename, timelapse_image)

# plot_mujoco_trajectories(dpc_traj, 'data/traj/dpc_traj_mujoco.png')
# plot_mujoco_trajectories(dpc_sf_traj, 'data/traj/dpc_sf_traj_mujoco.png')
# plot_mujoco_trajectories(vtnmpc_traj, 'data/traj/vtnmpc_traj_mujoco.png')
# plot_mujoco_trajectories(nmpc_traj, 'data/traj/nmpc_traj_mujoco.png')

def plot_mujoco_trajectories_video(traj, filename):
    quad_params = get_quad_params()
    state = quad_params["default_init_state_np"]
    Ti, Tf, Ts = 0.0, 20.0, 0.001
    desired_fps = 10

    # Total frames generated by the simulation
    total_frames = (Tf - Ti) / Ts

    # Frames required for a 60fps video over the entire duration
    frames_for_video = desired_fps * (Tf - Ti)

    # Calculate the divisor
    N = int(total_frames / frames_for_video)
    
    mj_quad = mujoco_quad(state=state, quad_params=quad_params, Ti=Ti, Tf=Tf, Ts=Ts, integrator='euler', render='mujoco')
    mj_quad.images = []
    traj = ptu.to_numpy(traj['X'][0])
    for i, x in enumerate(traj):
        if i % N == 0:
            mj_quad.set_state(x)
    video = np.stack(mj_quad.images)
    writer = imageio.get_writer(filename, fps=60)

    for i, img in enumerate(video):
        writer.append_data(img)

    # Close the writer to finalize the video file
    writer.close()

plot_mujoco_trajectories_video(dpc_traj, 'data/traj/dpc_traj_mujoco.mp4')
plot_mujoco_trajectories_video(dpc_sf_traj, 'data/traj/dpc_sf_traj_mujoco.mp4')
plot_mujoco_trajectories_video(vtnmpc_traj, 'data/traj/vtnmpc_traj_mujoco.mp4')
plot_mujoco_trajectories_video(nmpc_traj, 'data/traj/nmpc_traj_mujoco.mp4')

###### Video ######

# nav task
animator = Animator(
    states=             ptu.to_numpy(dpc_nav[2]['X'][0]),
    times=              np.arange(0,5,0.001),
    reference_history=  ptu.to_numpy(dpc_nav[2]['X'][0]),
    drawCylinder=True,
    elev=50,
    azim=-90,
    save_path='data/nav',
    save_name='dpc_nav',
    title='DPC Navigation'
)
animator.animate()

animator = Animator(
    states=             ptu.to_numpy(dpc_sf_nav[2]['X'][0]),
    times=              np.arange(0,5,0.001),
    reference_history=  ptu.to_numpy(dpc_sf_nav[2]['X'][0]),
    drawCylinder=True,
    elev=50,
    azim=-90,
    save_path='data/nav',
    save_name='dpc_sf_nav',
    title='DPC + PSF Navigation'
)
animator.animate()

animator = Animator(
    states=             ptu.to_numpy(vtnmpc_nav[2]['X'][0][1:]),
    times=              np.arange(0,5,0.001),
    reference_history=  ptu.to_numpy(vtnmpc_nav[2]['X'][0][1:]),
    drawCylinder=True,
    elev=50,
    azim=-90,
    save_path='data/nav',
    save_name='vtnmpc_nav',
    title='VTNMPC Navigation'
)
animator.animate()

animator = Animator(
    states=             ptu.to_numpy(nmpc_nav[2]['X'][0][1:]),
    times=              np.arange(0,5,0.001),
    reference_history=  ptu.to_numpy(nmpc_nav[2]['X'][0][1:]),
    drawCylinder=True,
    elev=50,
    azim=-90,
    save_path='data/nav',
    save_name='nmpc_nav',
    title='NMPC Navigation'
)
animator.animate()

# adv nav task
animator = Animator(
    states=             ptu.to_numpy(dpc_adv_nav[2]['X'][0]),
    times=              np.arange(0,10,0.001),
    reference_history=  ptu.to_numpy(dpc_adv_nav[2]['X'][0]),
    drawCylinder=True,
    elev=50,
    azim=-90,
    save_path='data/adv_nav',
    save_name='dpc_adv_nav',
    title='DPC Adversarial Navigation'
)
animator.animate()

animator = Animator(
    states=             ptu.to_numpy(dpc_sf_adv_nav[2]['X'][0]),
    times=              np.arange(0,10,0.001),
    reference_history=  ptu.to_numpy(dpc_sf_adv_nav[2]['X'][0]),
    drawCylinder=True,
    elev=50,
    azim=-90,
    save_path='data/adv_nav',
    save_name='dpc_sf_adv_nav',
    title='DPC + PSF Adversarial Navigation'
)
animator.animate()

animator = Animator(
    states=             ptu.to_numpy(vtnmpc_adv_nav[2]['X'][0][1:]),
    times=              np.arange(0,10,0.001),
    reference_history=  ptu.to_numpy(vtnmpc_adv_nav[2]['X'][0][1:]),
    drawCylinder=True,
    elev=50,
    azim=-90,
    save_path='data/adv_nav',
    save_name='vtnmpc_adv_nav',
    title='VTNMPC Adversarial Navigation'
)
animator.animate()

animator = Animator(
    states=             ptu.to_numpy(nmpc_adv_nav[2]['X'][0][1:]),
    times=              np.arange(0,10,0.001),
    reference_history=  ptu.to_numpy(nmpc_adv_nav[2]['X'][0][1:]),
    drawCylinder=True,
    elev=50,
    azim=-90,
    save_path='data/adv_nav',
    save_name='nmpc_adv_nav',
    title='NMPC Adversarial Navigation'
)
animator.animate()

# traj task
animator = Animator(
    states=             ptu.to_numpy(dpc_traj['X'][0][1:]),
    times=              np.arange(0,20,0.001),
    reference_history=  ptu.to_numpy(dpc_traj['R'][0][1:]),
    drawCylinder=False,
    elev=None,
    azim=None,
    save_path='data/traj',
    save_name='dpc_traj',
    title='DPC Trajectory Tracking'
)
animator.animate()

animator = Animator(
    states=             ptu.to_numpy(dpc_sf_traj['X'][0][1:]),
    times=              np.arange(0,20,0.001),
    reference_history=  ptu.to_numpy(dpc_sf_traj['R'][0][1:]),
    drawCylinder=False,
    elev=None,
    azim=None,
    save_path='data/traj',
    save_name='dpc_sf_traj',
    title='DPC + PSF Trajectory Tracking'
)
animator.animate()

animator = Animator(
    states=             ptu.to_numpy(vtnmpc_traj['X'][0][1:]),
    times=              np.arange(0,20,0.001),
    reference_history=  ptu.to_numpy(vtnmpc_traj['R'][0][1:]),
    drawCylinder=False,
    elev=None,
    azim=None,
    save_path='data/traj',
    save_name='vtnmpc_traj',
    title='VTNMPC Trajectory Tracking'
)
animator.animate()

animator = Animator(
    states=             ptu.to_numpy(nmpc_traj['X'][0][1:]),
    times=              np.arange(0,20,0.001),
    reference_history=  ptu.to_numpy(nmpc_traj['R'][0][1:]),
    drawCylinder=False,
    elev=None,
    azim=None,
    save_path='data/traj',
    save_name='nmpc_traj',
    title='NMPC Trajectory Tracking'
)
animator.animate()

print('fin')
