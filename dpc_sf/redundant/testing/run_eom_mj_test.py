""" 
This script previously showed that the mj was equivalent to the eom in
their bare bones fashion. I suspect the new eom is still correct as the 
MPC still works, but the mj does not, so I want to see if the new eom
is equivalent to the old mj, and if so then find out why new mj is diff
to old mj. Maybe it is a state read/write and changed behind the scenes
issue...
"""

from env import Sim
from quad import Quadcopter
from control.trajectory import waypoint_reference
import utils.pytorch_utils as ptu
import numpy as np
import torch

dt = 0.1
Ti = 0
Tf = 3.8
quad = Quadcopter()
backend = 'mj'
reference = waypoint_reference(type='wp_p2p', average_vel=1.6)

state = np.array([
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
])
if backend == 'mj':
    pass
elif backend == 'eom':
    state = ptu.from_numpy(state)

env = Sim(
    ### parameters for both backends
    dt=dt,
    Ti=Ti,
    Tf=Tf,
    params=quad.params,
    backend=backend,
    init_state=state,
    reference=reference,

    ### eom specific arguments
    state_dot=quad.state_dot,

    ### mj specific arguments
    xml_path="mujoco/quadrotor_x.xml",
    write_path="media/mujoco/",
)

if backend == 'eom':
    cmd = ptu.from_numpy(np.array([0.,0.,0.,0.]))
elif backend == 'mj':
    cmd = np.array([0.,0.,0.,0.])

state[15] = 523.3
env.set_state(state)
cmd[2] = 0

def command(cmd, t):
    cmd[2] = 1 * np.sin(t)
    return cmd

state_history = []
while env.t < env.Tf:
    state_history.append(env.state)
    print(command(cmd, env.t))
    env.step(command(cmd, env.t))
    
# Prepare combined data
if backend == 'eom':
    qpos_data_eom = ptu.to_numpy(torch.vstack(env.state_history)[:, 0:7])  # (x, y, z, q0, q1, q2, q3)
    qvel_data_eom = ptu.to_numpy(torch.vstack(env.state_history)[:, 7:13])  # (xdot, ydot, zdot, p, q, r)
elif backend == 'mj':
    qpos_data_eom = np.vstack(state_history)[:, 0:7]  # (x, y, z, q0, q1, q2, q3)
    qvel_data_eom = np.vstack(state_history)[:, 7:13]  # (xdot, ydot, zdot, p, q, r)

qpos_data_eom[-1,:] = qpos_data_eom[-2,:]
qvel_data_eom[-1,:] = qvel_data_eom[-2,:]

# Prepare labels
qpos_labels = ['x', 'y', 'z', 'q0', 'q1', 'q2', 'q3']
qvel_labels = ['xdot', 'ydot', 'zdot', 'p', 'q', 'r']

### MUJOCO SECTION ###

import numpy as np
import mediapy as media
import mujoco as mj
import os
import matplotlib.pyplot as plt

xml_path = "mujoco/quadrotor_x.xml"
write_path = "media/mujoco/"
d2r = np.pi / 180

# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model

# model geometry names
geom_names = [
    "arm_back0",
    "arm_front0",
    "arm_left0",
    "arm_right0",
    "core_geom",
    "floor",
]

# model body names
body_names = [
    "arm_back1",
    "arm_front1",
    "arm_left1",
    "arm_right1",
    "core",
    "thruster0",
    "thruster1",
    "thruster2",
    "thruster3",
    "world",
]

# mjdata constains the state and quantities that depend on it.
data = mj.MjData(model)

# the state is made up of time, generalised positions and velocities
time = data.time
qpos = data.qpos
qvel = data.qvel

# we can also retrieve cartesian coordinates, however this is zeros until
# the simulation is propogated once
xpos = data.geom_xpos

# propogate the simulation to get non generalised state data with minimal function
mj.mj_kinematics(model, data)

# MjData also supports named access:
# arm_front0_xpos = data.geom('arm_front0').xpos

# Make renderer, render and show the pixels
# model.vis.scale.framelength *= 10
# model.vis.scale.framewidth *= 10
renderer = mj.Renderer(model=model, height=720, width=1280)

# the image will still be black
image = renderer.render()
media.write_image(write_path + "image.png", image)

# the minimal function doesnt invoke entire pipeline so a rendered image will
# still be black as if no dynamics propogated, therefore use full mj_forward
mj.mj_forward(model, data)
renderer.update_scene(data)

# now we will get an image of the scene
media.write_image(write_path + "image.png", renderer.render())

### lets simulate and render a video
duration = 3.8  # (seconds)
framerate = 60  # (Hz)

# Simulate and display video.
frames = []
qpos_data = []
qvel_data = []
mj.mj_resetData(model, data)  # Reset state and time.
data.ctrl = [0.0, 0.0, 0.0, 0.0]
data.ctrl = [2.943]*4

omega = 523.3
data.ctrl[2] = 1.076e-5*(omega)**2 # take one motor to 80 % causing a crash

while data.time < duration:
    # kTh * w_hover ** 2 = 2.943
    # data.ctrl = [3]*4

    mj.mj_step(model, data)
    data.ctrl[2] = 1.076e-5*(omega)**2 # take one motor to 80 % causing a crash
    omega += command(cmd, data.time)[2]/quad.params['IRzz'] * dt

    qpos_data.append(data.qpos.copy()) # copy is necessary otherwise it'll be overwritten
    qvel_data.append(data.qvel.copy())
    if len(frames) < data.time * framerate:

        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(pixels)
media.write_video(write_path + "video.mp4", frames, fps=framerate)

# def mjstate2mpcstate(mjstate):
#     mpcstate = mjstate.copy()
#     flipped_state_idx = [1,2,5,8,9,11]
#     mpcstate[:,flipped_state_idx] *= -1
#     return mpcstate

# Convert list of arrays to 2D array
qpos_data = np.vstack(qpos_data)
qvel_data = np.vstack(qvel_data)

qpos_data[:,1] *= -1
qpos_data[:,2] *= -1
qpos_data[:,5] *= -1

qvel_data[:,1] *= -1
qvel_data[:,2] *= -1
qvel_data[:,4] *= -1

qpos_data_mj = qpos_data
qvel_data_mj = qvel_data

# Plotting
# Create subplots for qpos_data
plt.figure(figsize=(5, 8))  # Adjust the figure size as needed
for i in range(7):
    plt.subplot(7, 1, i + 1)
    plt.plot(qpos_data_eom[:, i], label='eom')
    plt.plot(qpos_data_mj[:, i], label='mj')
    plt.legend()

    plt.ylabel(qpos_labels[i])

plt.xlabel('Time step')
plt.suptitle('QPos over time')
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Prevent the title from overlapping

# Create subplots for qvel_data
plt.figure(figsize=(5, 8))  # Adjust the figure size as needed
for i in range(6):
    plt.subplot(6, 1, i + 1)
    plt.plot(qvel_data_eom[:, i], label='eom')
    plt.plot(qvel_data_mj[:, i], label='mj')
    plt.legend()

    plt.ylabel(qvel_labels[i])

plt.xlabel('Time step')
plt.suptitle('QVel over time')
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Prevent the title from overlapping

plt.show()