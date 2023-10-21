from quad import Quadcopter
from utils.windModel import Wind
from utils.stateConversions import sDes2state
from trajectory import Trajectory
from trajectory_rework import waypoint_reference
# from mpc import MPC_Point_Ref, MPC_Point_Ref_Obstacle, MPC_Traj_Ref
from mpc import MPC_Point_Ref, MPC_Point_Ref_Obstacle, MPC_Traj_Ref
import torch
import time
import numpy as np
import utils
import matplotlib.pyplot as plt

start_time = time.time()

# Simulation Setup
# --------------------------- 
Ti = 0
Ts = 0.1
interaction_interval = 1
ctrlHzn = 30 # 3 seconds good for p2p and trajref
Tf = 5 # 50 seconds end of point2point
ifsave = True

# Choose type of simulation (only make one True)
p2p = False # point 2 point
trajref = False
obsavoid = True # point 2 point, but avoid an obstacle on the way

# Choose trajectory settings
# --------------------------- 
ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]
trajSelect = np.zeros(3)

# Select Control Type             (0: xyz_pos,                  1: xy_vel_z_pos,            2: xyz_vel)
ctrlType = ctrlOptions[0]   
# Select Position Trajectory Type (0: hover,                    1: pos_waypoint_timed,      2: pos_waypoint_interp,    
#                                  3: minimum velocity          4: minimum accel,           5: minimum jerk,           6: minimum snap
#                                  7: minimum accel_stop        8: minimum jerk_stop        9: minimum snap_stop
#                                 10: minimum jerk_full_stop   11: minimum snap_full_stop
#                                 12: pos_waypoint_arrived     13: pos_waypoint_arrived_wait
if p2p: # for point 2 point
    trajSelect[0] = 13 # use 13 for point2point
    # Select Yaw Trajectory Type      (0: none                      1: yaw_waypoint_timed,      2: yaw_waypoint_interp     3: follow          4: zero)
    trajSelect[1] = 3 # 3 is good       
    # Select if waypoint time is used, or if average speed is used to calculate waypoint time   (0: waypoint time,   1: average speed)
    trajSelect[2] = 0 # 0 for point2point     
elif trajref: # for trajectory reference
    trajSelect[0] = 2 # use 13 for point2point
    # Select Yaw Trajectory Type      (0: none                      1: yaw_waypoint_timed,      2: yaw_waypoint_interp     3: follow          4: zero)
    trajSelect[1] = 3 # 3 is good       
    # Select if waypoint time is used, or if average speed is used to calculate waypoint time   (0: waypoint time,   1: average speed)
    trajSelect[2] = 1 # 0 for point2point     
elif obsavoid:
    trajSelect[0] = 13 # use 13 for point2point
    # Select Yaw Trajectory Type      (0: none                      1: yaw_waypoint_timed,      2: yaw_waypoint_interp     3: follow          4: zero)
    trajSelect[1] = 3 # 3 is good       
    # Select if waypoint time is used, or if average speed is used to calculate waypoint time   (0: waypoint time,   1: average speed)
    trajSelect[2] = 0 # 0 for point2point  
else: 
    raise KeyError("invalid task selection")
print("Control type: {}".format(ctrlType))

# Initialize Quadcopter, Controller, Wind, Result Matrixes
# ---------------------------
quad = Quadcopter()
traj = Trajectory(quad, ctrlType, trajSelect)

reference = waypoint_reference(type='wp_p2p', average_vel=1)


if p2p:
    ctrl = MPC_Point_Ref(N=ctrlHzn, sim_Ts=Ts, interaction_interval=interaction_interval, n=17, m=4, quad=quad)
elif trajref:
    ctrl = MPC_Traj_Ref(N=ctrlHzn, sim_Ts=Ts, interaction_interval=interaction_interval, n=17, m=4, quad=quad, reference_traj=traj)
elif obsavoid:
    ctrl = MPC_Point_Ref_Obstacle(N=ctrlHzn, sim_Ts=Ts, interaction_interval=interaction_interval, n=17, m=4, quad=quad)
    
wind = Wind('None', 2.0, 90, -15)

# Trajectory for First Desired States
# ---------------------------
sDes = traj.desiredState(0, Ts, quad)

def get_reference(t):
    sDes = traj.desiredState(t+5, Ts, quad)   
    state = utils.stateConversions.sDes2state(sDes)
    state_test = np.array(
        [2.00000000e+00, 2.00000000e+00, 1.00000000e+00, 9.23879528e-01,
        3.82683442e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 5.22984741e+02, 5.22984741e+02, 5.22984741e+02,
        5.22984741e+02]
    )  
    print(f'delta: {state-state_test}') 
    return state

if p2p:
    reference = sDes2state(sDes)
    cmd = ctrl(quad.state.tolist(), reference.tolist()).value(ctrl.U)[:,0]
elif trajref:
    # trajectory following uses a-priori knowledge of the trajectory
    cmd = ctrl(quad.state.tolist(), Ti).value(ctrl.U)[:,0]
elif obsavoid:
    reference = sDes2state(sDes)
    cmd = ctrl(quad.state.tolist(), reference.tolist()).value(ctrl.U)[:,0]


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
for i, t in enumerate(np.arange(Ti,Tf,Ts)):

    print(i)
    # Dynamics (using last timestep's commands)
    # ---------------------------
    quad.update(cmd, wind, t, Ts)

    # Trajectory for Desired States 
    # ---------------------------
    # this is for point2point
    if p2p:
        sDes = traj.desiredState(t+5, Ts, quad)     
        reference = sDes2state(sDes) 
        print(f'state error: {quad.state - reference}')
        print(f'input: {cmd}')
    # we just use traj directly for traj_ref
    elif trajref:
        print(f'state error: {quad.state - sDes2state(traj.desiredState(t, Ts, quad))}')
        print(f'input: {cmd}')
    elif obsavoid:
        sDes = traj.desiredState(t+5, Ts, quad)     
        reference = sDes2state(sDes) 
        reference = get_reference(t)
        print(f'state error: {quad.state - reference}')
        print(f'input: {cmd}')

    # Generate Commands (for next iteration)
    # ---------------------------
    if i % interaction_interval == 0:
        print(f'{i} is a multiple of {interaction_interval}')
        # this is for point2point
        if p2p:
            cmd = ctrl(quad.state.tolist(), reference.tolist()).value(ctrl.U)[:,0]
        # this is for traj_ref
        elif trajref:
            cmd = ctrl(quad.state.tolist(), t).value(ctrl.U)[:,0]
        elif obsavoid:
            cmd = ctrl(quad.state.tolist(), reference.tolist()).value(ctrl.U)[:,0]
    
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

    

end_time = time.time()
print("Simulated {:.2f}s in {:.6f}s.".format(t, end_time - start_time))

# View Results
# ---------------------------

# utils.fullprint(sDes_traj_all[:,3:6])
# utils.makeFigures(quad.params, t_all, pos_all, vel_all, quat_all, omega_all, euler_all, w_cmd_all, wMotor_all, thr_all, tor_all, sDes_traj_all, sDes_calc_all)
#ani = utils.sameAxisAnimation(t_all, traj.wps, pos_all, quat_all, sDes_traj_all, Ts, quad.params, traj.xyzType, traj.yawType, ifsave)
if p2p or trajref:
    animator = utils.Animator(t_all, traj.wps, pos_all, quat_all, sDes_traj_all, Ts, quad.params, traj.xyzType, traj.yawType, ifsave, drawCylinder=False)
elif obsavoid:
    animator = utils.Animator(t_all, traj.wps, pos_all, quat_all, sDes_traj_all, Ts, quad.params, traj.xyzType, traj.yawType, ifsave, drawCylinder=True)
ani = animator.animate()
plt.show()