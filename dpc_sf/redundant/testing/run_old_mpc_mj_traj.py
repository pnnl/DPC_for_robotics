import numpy as np
import mediapy as media
import mujoco as mj
import os
from redundant.testing.quad import sys_params, Quadcopter
from redundant.testing.mpc import MPC_Point_Ref_Obstacle
from redundant.testing.trajectory import Trajectory
from utils.stateConversions import sDes2state

# NEW REFERENCE DOES NOT WORK ON MUJOCO
# -------------------------------------
# testing trajectory being incorrect
#from trajectory import waypoint_reference
#reference = waypoint_reference(type='wp_p2p', average_vel=0.5)
# -------------------------------------

interaction_interval = 1
sim_Ts = 0.1

# quadcopter parameters
params,_,_ = sys_params()

# mpc
quad = Quadcopter()
ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]
ctrlType = ctrlOptions[0]   
trajSelect = np.zeros(3)
trajSelect[0] = 2 # use 13 for point2point
trajSelect[1] = 3 # 3 is good       
trajSelect[2] = 1 # 0 for point2point  
traj = Trajectory(quad, ctrlType, trajSelect)
traj = Trajectory(quad, 'xyz_pos', np.array([13,3,0]))
ctrl = MPC_Point_Ref_Obstacle(N=30, sim_Ts=sim_Ts, interaction_interval=interaction_interval, n=17, m=4, quad=quad)

xml_path = "mujoco/quadrotor_x.xml"
write_path = "media/mujoco/"
d2r = np.pi / 180

# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
dt = model.opt.timestep # should be 0.01
assert sim_Ts == dt
assert model.opt.integrator == 0 # 0 == semi implicit euler, closest to explicit euler mj offers

# mjdata constains the state and quantities that depend on it.
data = mj.MjData(model)

# Make renderer, render and show the pixels
# model.vis.scale.framelength *= 10
# model.vis.scale.framewidth *= 10
renderer = mj.Renderer(model=model, height=720, width=1280)

### lets simulate and render a video
duration = 3.8  # (seconds)
framerate = 60  # (Hz)

# define the actuator dynamics alone
def actuator_dot(cmd):
    return cmd/params["IRzz"]

# Simulate and display video.
frames = []
mj.mj_resetData(model, data)  # Reset state and time.
data.ctrl = [params["kTh"] * params["w_hover"] ** 2] * 4 # kTh * w_hover ** 2 = 2.943
omegas = np.array([params["w_hover"]]*4)
i = 0
t = 0

# need to call this at time = 0 to instantiate some attributes
sDes = traj.desiredState(0, dt, quad)

def get_state_from_mj(data):
    # generalised positions/velocities not in right coordinates
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    qpos[1] *= -1
    qpos[2] *= -1
    qpos[5] *= -1

    qvel[1] *= -1
    qvel[2] *= -1 # 
    qvel[4] *= -1 # qdot

    return np.concatenate([qpos, qvel, omegas]).flatten()

while data.time < duration:

    state = get_state_from_mj(data)

    # define reference
    sDes = traj.desiredState(t+5, dt, quad)     
    reference = sDes2state(sDes) 
    print(f'state error: {state - reference}')

    # choose command based on state
    if i % interaction_interval == 0:
        try:
            cmd = ctrl(state.tolist(), reference.tolist()).value(ctrl.U)[:,0]
        except:
            # create video up to point of infeasibility to debug
            media.write_video(write_path + "video.mp4", frames, fps=framerate)
    # cmd = np.array([np.sin(data.time)*0.001,0,np.sin(data.time)*0.001,0])
    print(f'input: {cmd}')

    # translate omegas to thrust (mj input)
    thr = params["kTh"] * omegas ** 2
    data.ctrl = thr.tolist()
    print(data.time)
    print(data.ctrl)

    # update mujoco and actuators with EULER
    mj.mj_step(model, data)
    omegas += actuator_dot(cmd) * dt

    # update counter
    i += 1
    t += dt

    # draw
    if len(frames) < data.time * framerate:
        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(pixels)

media.write_video("video.mp4", frames, fps=framerate)

