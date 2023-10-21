from quad import Quadcopter
from trajectory import Trajectory
import numpy as np
import utils
import matplotlib.pyplot as plt
from utils.windModel import Wind

### EOM SECTION ###

quad = Quadcopter()
traj = Trajectory(quad, "xyz_pos", np.array([13,3,0]))
wind = Wind('None', 2.0, 90, -15)
ifsave = True

Tf = 3.8
Ts = 0.1
Ti = 0

# Initialize Result Matrixes
# ---------------------------
numTimeStep = int(Tf/Ts+1)

t_all          = np.zeros(numTimeStep)
s_all          = np.zeros([numTimeStep, len(quad.state)])
pos_all        = np.zeros([numTimeStep, len(quad.pos)])
vel_all        = np.zeros([numTimeStep, len(quad.vel)])
quat_all       = np.zeros([numTimeStep, len(quad.quat)])
omega_all      = np.zeros([numTimeStep, len(quad.omega)])
euler_all      = np.zeros([numTimeStep, len(quad.euler)])
sDes_traj_all  = np.zeros([numTimeStep, len(traj.sDes)])
# sDes_calc_all  = np.zeros([numTimeStep, len(ctrl.sDesCalc)])
# w_cmd_all      = np.zeros([numTimeStep, len(ctrl.w_cmd)])
wMotor_all     = np.zeros([numTimeStep, len(quad.wMotor)])
thr_all        = np.zeros([numTimeStep, len(quad.thr)])
tor_all        = np.zeros([numTimeStep, len(quad.tor)])

t_all[0]            = Ti
s_all[0,:]          = quad.state
pos_all[0,:]        = quad.pos
vel_all[0,:]        = quad.vel
quat_all[0,:]       = quad.quat
omega_all[0,:]      = quad.omega
euler_all[0,:]      = quad.euler
sDes_traj_all[0,:]  = traj.sDes
# sDes_calc_all[0,:]  = ctrl.sDesCalc
# w_cmd_all[0,:]      = ctrl.w_cmd
wMotor_all[0,:]     = quad.wMotor
thr_all[0,:]        = quad.thr
tor_all[0,:]        = quad.tor

# Run Simulation
# ---------------------------
t = Ti
i = 1
# Generate Commands (for next iteration)
# ---------------------------
cmd = np.array([0,0,0,0])
quad.state[15] = 523.3
# quad.state[13:] = np.sqrt(3/1.076e-5) # test weight
for i, t in enumerate(np.arange(Ti,Tf,Ts)):

    # Dynamics (using last timestep's commands)
    # ---------------------------
    state = quad.update(cmd, wind, t, Ts)

    # print("{:.3f}".format(t))
    t_all[i]             = t
    s_all[i,:]           = quad.state
    pos_all[i,:]         = quad.pos
    vel_all[i,:]         = quad.vel
    quat_all[i,:]        = quad.quat
    omega_all[i,:]       = quad.omega
    euler_all[i,:]       = quad.euler
    sDes_traj_all[i,:]   = traj.sDes
    # sDes_calc_all[i,:]   = ctrl.sDesCalc
    # w_cmd_all[i,:]       = ctrl.w_cmd
    wMotor_all[i,:]      = quad.wMotor
    thr_all[i,:]         = quad.thr
    tor_all[i,:]         = quad.tor

# Extract required data
pos_data = pos_all
quat_data = quat_all
vel_data = vel_all
omega_data = omega_all

# Prepare combined data
qpos_data_eom = np.hstack([pos_data, quat_data])  # (x, y, z, q0, q1, q2, q3)
qvel_data_eom = np.hstack([vel_data, omega_data])  # (xdot, ydot, zdot, p, q, r)
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

xml_path = "../../mujoco/quadrotor_x.xml"
write_path = "../../media/mujoco/"
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
while data.time < duration:
    # kTh * w_hover ** 2 = 2.943
    data.ctrl = [2.943]*4
    data.ctrl[2] = 1.076e-5*(523.3)**2 # take one motor to 80 % causing a crash
    # data.ctrl = [3]*4
    mj.mj_step(model, data)
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
    plt.plot(qpos_data_eom[:, i])
    plt.plot(qpos_data_mj[:, i])

    plt.ylabel(qpos_labels[i])

plt.xlabel('Time step')
plt.suptitle('QPos over time')
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Prevent the title from overlapping

# Create subplots for qvel_data
plt.figure(figsize=(5, 8))  # Adjust the figure size as needed
for i in range(6):
    plt.subplot(6, 1, i + 1)
    plt.plot(qvel_data_eom[:, i])
    plt.plot(qvel_data_mj[:, i])
    plt.ylabel(qvel_labels[i])

plt.xlabel('Time step')
plt.suptitle('QVel over time')
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Prevent the title from overlapping

plt.show()