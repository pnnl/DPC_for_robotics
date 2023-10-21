# in order to check that the waypoint based trajectory generation of the rework
# is correct I will sample both here for a number of points and test to see if
# they are the same

# shared imports
import numpy as np
import utils

# import and setup original
from trajectory import Trajectory
from quad import Quadcopter
from utils.windModel import Wind

quad = Quadcopter()

# trajectory: np.array([2,3,1])
# p2p: np.array([13,3,0])
traj = Trajectory(quad, 'xyz_pos', np.array([13,3,0]))
# instantiate some attributes:
sDes = traj.desiredState(0, 0.1, quad)
reference0 = utils.stateConversions.sDes2state(sDes)

# import rework
from trajectory_rework import waypoint_reference
wp_ref = waypoint_reference(
    type='wp_p2p',
    average_vel=1
)

def get_original_ref(t):
    sDes = traj.desiredState(t, 0.1, quad)
    return utils.stateConversions.sDes2state(sDes)

def get_rework_ref(t):
    return wp_ref(t)

def print_delta(t):
    print(f'delta is: {get_original_ref(t) - get_rework_ref(t)}')

print('f')