import dpc_sf.control.rl.gym_environments.multirotor_utils as utils
import numpy as np
from dpc_sf.dynamics.params import params
from dpc_sf.utils.rotationConversion import YPRToQuat_np, quatToYPR_ZYX_np

def random_state(env_bounding_box=4.0, init_max_attitude=np.pi/3.0, init_max_vel=0.5, init_max_angular_vel=0.1*np.pi, include_actuators=True):
    desired_position = np.zeros(3)
    # attitude (roll pitch yaw)
    quat_init = np.array([1., 0., 0., 0.])
    attitude_euler_rand = np.random.uniform(low=-init_max_attitude, high=init_max_attitude, size=(3,))
    # quat_init = utils.euler2quat(attitude_euler_rand)
    # lets use the other euler2quat
    # this is an equivalent transformation:
    quat_init = YPRToQuat_np(attitude_euler_rand[2], attitude_euler_rand[1], attitude_euler_rand[0])
    assert np.abs(np.flip(quatToYPR_ZYX_np(quat_init)) - attitude_euler_rand).max() <= 1e-8

    # position (x, y, z)
    c = 0.2
    ep = np.random.uniform(low=-(env_bounding_box-c), high=(env_bounding_box-c), size=(3,))
    pos_init = ep + desired_position
    # velocity (vx, vy, vz)
    vel_init = utils.sample_unit3d() * init_max_vel
    # angular velocity (wx, wy, wz)
    angular_vel_init = utils.sample_unit3d() * init_max_angular_vel
    # omegas 
    if include_actuators:
        omegas_init = np.array([params['w_hover']]*4)
        state_init = np.concatenate([pos_init, quat_init, vel_init, angular_vel_init, omegas_init]).ravel()
    else:
        state_init = np.concatenate([pos_init, quat_init, vel_init, angular_vel_init]).ravel()
    return state_init