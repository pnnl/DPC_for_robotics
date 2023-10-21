from quad import Quadcopter
from utils.windModel import Wind
from utils.stateConversions import sDes2state
from trajectory import Trajectory
# from mpc import MPC_Point_Ref, MPC_Point_Ref_Obstacle, MPC_Traj_Ref
from mpc import MPC_Point_Ref, MPC_Point_Ref_Obstacle, MPC_Traj_Ref
import torch
import time
import numpy as np
import utils
import matplotlib.pyplot as plt
from quad_gym import QuadcopterGym

start_time = time.time()

# Simulation Setup
# --------------------------- 
Ti = 0
Ts = 0.01
interaction_interval = 10
ctrlHzn = 30 # 3 seconds good for p2p and trajref
Tf = 30 # 50 seconds end of point2point
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
wind = Wind('None', 2.0, 90, -15)

# Trajectory for First Desired States
# ---------------------------
sDes = traj.desiredState(0, Ts, quad)

# instantiate the gym_env
env = QuadcopterGym(quad=quad)
