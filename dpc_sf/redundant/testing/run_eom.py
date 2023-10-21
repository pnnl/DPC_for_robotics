from quad import Quadcopter
from trajectory import Trajectory
import numpy as np
import utils
import matplotlib.pyplot as plt
from utils.windModel import Wind

quad = Quadcopter()
traj = Trajectory(quad, "xyz_pos", np.array([13,3,0]))
wind = Wind('None', 2.0, 90, -15)
ifsave = True

Tf = 3.8
Ts = 0.01
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
quad.state[13] = 523.3
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
qpos_data = np.hstack([pos_data, quat_data])  # (x, y, z, q0, q1, q2, q3)
qvel_data = np.hstack([vel_data, omega_data])  # (xdot, ydot, zdot, p, q, r)

# Prepare labels
qpos_labels = ['x', 'y', 'z', 'q0', 'q1', 'q2', 'q3']
qvel_labels = ['xdot', 'ydot', 'zdot', 'p', 'q', 'r']

# Plotting
# Create subplots for qpos_data
plt.figure(figsize=(5, 8))  # Adjust the figure size as needed
for i in range(7):
    plt.subplot(7, 1, i + 1)
    plt.plot(qpos_data[:, i])
    plt.ylabel(qpos_labels[i])

plt.xlabel('Time step')
plt.suptitle('QPos over time')
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Prevent the title from overlapping

# Create subplots for qvel_data
plt.figure(figsize=(5, 8))  # Adjust the figure size as needed
for i in range(6):
    plt.subplot(6, 1, i + 1)
    plt.plot(qvel_data[:, i])
    plt.ylabel(qvel_labels[i])

plt.xlabel('Time step')
plt.suptitle('QVel over time')
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Prevent the title from overlapping

plt.show()


# View Results
# ---------------------------
# animator = utils.Animator(t_all, traj.wps, pos_all, quat_all, sDes_traj_all, Ts, quad.params, traj.xyzType, traj.yawType, ifsave, drawCylinder=False)
# ani = animator.animate()
# plt.show()