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
import copy

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

traj = waypoint_reference(type='wp_p2p', average_vel=1)


if p2p:
    ctrl = MPC_Point_Ref(N=ctrlHzn, sim_Ts=Ts, interaction_interval=interaction_interval, n=17, m=4, quad=quad)
elif trajref:
    ctrl = MPC_Traj_Ref(N=ctrlHzn, sim_Ts=Ts, interaction_interval=interaction_interval, n=17, m=4, quad=quad, reference_traj=traj)
elif obsavoid:
    ctrl = MPC_Point_Ref_Obstacle(N=ctrlHzn, sim_Ts=Ts, interaction_interval=interaction_interval, n=17, m=4, quad=quad)
    
wind = Wind('None', 2.0, 90, -15)

# Trajectory for First Desired States
# ---------------------------

def get_reference(t):
    sDes = traj.desiredState(t+5, Ts, quad)     
    return utils.stateConversions.sDes2state(sDes) 

reference = traj(0)
cmd = ctrl(quad.state.tolist(), reference.tolist()).value(ctrl.U)[:,0]


# Initialize Result Matrixes
# ---------------------------
# we need to create a saving attribute to preserve state
state_history = []
time_history = []
reference_history = []

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
    # save current state to history
    state_history.append(copy.deepcopy(quad.state)) # deepcopy required, tensor stored by reference
    time_history.append(t) # no copy required as it is a float, not stored by reference
    reference_history.append(np.copy(traj(t))) # np.copy great as reference is a np.array

    print(f'state error: {quad.state - reference}')
    print(f'input: {cmd}')

    # Generate Commands (for next iteration)
    # ---------------------------
    if i % interaction_interval == 0:
        print(f'{i} is a multiple of {interaction_interval}')
        cmd = ctrl(quad.state.tolist(), reference.tolist()).value(ctrl.U)[:,0]


    

end_time = time.time()
print("Simulated {:.2f}s in {:.6f}s.".format(t, end_time - start_time))

# View Results
# ---------------------------
from utils.animation_rework import Animator
import utils.pytorch_utils as ptu
animator = Animator(
    states=ptu.to_numpy(torch.vstack(state_history)), 
    times=np.array(time_history), 
    reference_history=np.vstack(reference_history), 
    reference=traj, 
    reference_type='wp_p2p', 
    drawCylinder=True
)
animator.animate() # contains the plt.show()

plt.show()