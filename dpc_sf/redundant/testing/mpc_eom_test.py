# In this script I will assess the MPC on state/reference pairs it has failed on before,
# saved in states/

import numpy as np
import os
from control.mpc import MPC_Point_Ref_Obstacle as MPC_Point_Ref_Obstacle_2
from control.mpc import MPC_Point_Ref as MPC_Point_Ref_2
from redundant.testing.mpc import MPC_Point_Ref_Obstacle as MPC_Point_Ref_Obstacle_1
from redundant.testing.mpc import MPC_Point_Ref as MPC_Point_Ref_1
from quad import Quadcopter as QuadcopterRF
from redundant.testing.quad import Quadcopter
import matplotlib.pyplot as plt
from utils.animation import Animator

# load data
def load_data(directory):
    data = []
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            data.append(np.load(file_path))
    return data

# just choose latest datapoint
data = load_data('states')
state = data[-1]['state']
ref = data[-1]['reference']

# load quad for its parameters
quad_1 = Quadcopter()
quad_2 = QuadcopterRF()

# setup controllers
ctrl_obs_1 = MPC_Point_Ref_Obstacle_1(
    N=30,
    sim_Ts=0.1,
    interaction_interval=1, 
    n=17,
    m=4,
    quad=quad_1
)

ctrl_obs_2 = MPC_Point_Ref_Obstacle_2(
    N=30,
    dt=0.1,
    interaction_interval=1, 
    n=17, 
    m=4, 
    dynamics=quad_2.casadi_state_dot,
    state_ub=quad_2.params["ca_state_ub"],
    state_lb=quad_2.params["ca_state_lb"],
    return_type='torch'
)
ctrl_1 = MPC_Point_Ref_1(
    N=30,
    sim_Ts=0.1,
    interaction_interval=1, 
    n=17,
    m=4,
    quad=quad_1
)

ctrl_2 = MPC_Point_Ref_2(
    N=30,
    dt=0.1,
    interaction_interval=1, 
    n=17, 
    m=4, 
    dynamics=quad_2.casadi_state_dot,
    state_ub=quad_2.params["ca_state_ub"],
    state_lb=quad_2.params["ca_state_lb"],
    return_type='torch'
)

# this will fail, so then we can debug
try:
    ctrl_obs_2(state, ref)
except:
    pass

# this can be used to change the tolerance on the fly
# ctrl_obs_2.opti.solver('ipopt', {
#             'ipopt.print_level':0, 
#             'print_time':0,
#             'ipopt.tol': 1e-4
#         })

state_prediction = ctrl_obs_2.opti.debug.value(ctrl_obs_2.X)
input_prediction = ctrl_obs_2.opti.debug.value(ctrl_obs_2.U)

nominal_points = []
violations = []
for idx, state_p in enumerate(state_prediction.T):
    if ctrl_obs_2.is_in_cylinder(state_p[0], state_p[1], 1, 1, 0.5):
        print('entering cylinder')
    else:
        print('nominal')
        print(state_p)

radius = 0.5
center = (1, 1)

theta = np.linspace(0, 2 * np.pi, 100)  # Angle values from 0 to 2*pi
x = center[0] + radius * np.cos(theta)  # x-coordinates of the circle points
y = center[1] + radius * np.sin(theta)  # y-coordinates of the circle points

plt.figure()
plt.plot(x, y)

plt.plot(state_prediction[0,:], state_prediction[1,:], label='predicted state history')

plt.scatter(state[0], state[1], marker='x', color='green', label='true start location')
plt.scatter(state_prediction[0,0], state_prediction[1,0], marker='x', color='red', label='predicted start location')

plt.axis('equal')  # Equal aspect ratio for x and y axes
plt.xlabel('X')
plt.ylabel('Y')
plt.title('')
plt.legend()
plt.grid(True)
plt.show()

# plot 3D line
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the line
ax.plot(state_prediction[0,:], state_prediction[1,:], state_prediction[2,:])
ax.scatter(state_prediction[0,0], state_prediction[1,0], state_prediction[2,0], marker='x', color='red', label='predicted start point')
ax.scatter(state[0], state[1], state[2], marker='x', color='green', label='actual start point')
ax.scatter(ref[0], ref[1], ref[2], marker='x', color='black', label='reference point')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

# animate if needed
from control.trajectory import waypoint_reference
reference = waypoint_reference(type='wp_p2p', average_vel=0.5)

animator = Animator(
    states=state_prediction, 
    times=np.linspace(0,0.1*30,31), 
    reference_history=np.vstack([ref]*31).T, 
    reference=reference, 
    reference_type='wp_p2p', 
    drawCylinder=True
)
animator.animate() # contains the plt.show()

print(data)