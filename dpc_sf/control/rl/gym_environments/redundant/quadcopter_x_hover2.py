"""
This file uses the true state: and it WORKS
"""

import numpy as np
from dpc_sf.gym_environments.base_env import UAVBaseEnv

import dpc_sf.gym_environments.multirotor_utils as utils

from typing import Optional

import copy
from gymnasium import spaces
from dpc_sf.dynamics.params import params
from dpc_sf.gym_environments import multirotor_utils


class QuadrotorXHoverEnv(UAVBaseEnv):
    """
    Quadrotor with plus(+) configuration.
    Environment designed to make the UAV hover at the desired position.

    * Environment Name: QuadrotorPlusHoverEnv-v0

    Args:
        frame_skip (int): Number of frames to skip before application of next action command to the environment from the control policy.
    """

    def __init__(
            self, 
            state,
            reference,
            Q,
            R,
            env_bounding_box=1.2, 
            randomize_reset=False
        ):
        super().__init__(
            state=state,
            reference=reference,
            env_bounding_box=env_bounding_box, 
            randomize_reset=randomize_reset
        )

        sa_dtype = np.float64

        self.action_space = spaces.Box(low=np.array([-1,-1,-1,-1]), high=np.array([1,1,1,1], dtype=sa_dtype))

        # I can only say this is the obs space if the actual returned observation is a dict much like this
        f32_state_ub = copy.deepcopy(params['state_ub']).astype(sa_dtype)
        f32_state_lb = copy.deepcopy(params['state_lb']).astype(sa_dtype)
        # f32_cyl_ub = np.array([1.5, 1.5, 0.5], dtype=sa_dtype)
        # f32_cyl_lb = np.array([0.5, 0.5, 0.2], dtype=sa_dtype)

        # must instead stack the reference and the observation
        # [e_pos, rot_mat.flatten(), vel, angular_vel]
        self.observation_space = spaces.Box(
            low = f32_state_lb, 
            high = f32_state_ub, 
            dtype=sa_dtype
        )
        

    @property
    def hover_force(self):
        """
        Hover force for each actuators in quadcopter.

        Returns:
            float: Hover force for each rotor.
        """
        return self.mass * self.gravity_mag * 0.25

    def get_motor_input(self, action):
        """
        Transform policy actions to motor inputs.

        Args:
            action (numpy.ndarray): Actions from policy of shape (4,).

        Returns:
            numpy.ndarray: Vector of motor inputs of shape (4,).
        """
        # the below allows the RL to specify omegas between 122.98 and 922.98
        # the true range is 75 - 925, but this is just a quick and dirty translation
        # and keeps things symmetrical about the hover omega

        motor_range = 400.0 # 400.0 # 2.0
        hover_w = self.quad.params['w_hover']
        cmd_w = hover_w + action * motor_range / (self.policy_range[1] - self.policy_range[0])
        
        # the above is the commanded omega
        w_error = cmd_w - self.quad.get_state()[13:]
        p_gain = self.quad.params["IRzz"] / self.quad.Ts

        motor_inputs = w_error * p_gain
        
        return motor_inputs

    def step(self, action):
        """
        Method to take an action for agent transition in the environment

        Args:
            action (numpy.ndarray): Action vector. Expects each element of the action vector between [-1., 1.].

        Returns:
            tuple[numpy.ndarray, float, bool, dict]: Output tuple contains the following elements in the given order:
                - ob (numpy.ndarray): Observation vector.
                - reward (float): Scalar reward value.
                - done (bool): ``True`` if episode is over else ``False``.
                - info (dict): Dictionary of additional information from simulation if any.
        """

        self._time += 1
        a = self.clip_action(action, a_min=self.policy_range[0], a_max=self.policy_range[1])
        action_mujoco = self.get_motor_input(a)

        self.quad.step(action_mujoco)


        ob = self._get_obs()
        self.current_robot_observation = ob.copy()

        reward, reward_info = self.get_reward(ob, a)

        info = {"reward_info": reward_info,
                "desired_goal": self.desired_position.copy(),
                "mujoco_qpos": self.mujoco_qpos,
                "mujoco_qvel": self.mujoco_qvel}

        done = self.is_done(ob)

        if self.observation_noise_std:
            ob += self.np_random.uniform(low=-self.observation_noise_std, high=self.observation_noise_std, size=ob.shape)
        return ob, reward, done, False, info

    def _get_obs(self):
        """
        Full observation of the environment.

        Returns:
            numpy.ndarray: 18-dim numpy array of states of environment consisting of (err_x, err_y, err_z, rot_mat(3, 3), vx, vy, vz, body_rate_x, body_rate_y, body_rate_z)
        """
        state = self.quad.get_state()
        qpos = state[0:7] # self.sim.data.qpos.copy()
        qvel = state[7:13] # self.sim.data.qvel.copy()

        self.mujoco_qpos = np.array(qpos)
        self.mujoco_qvel = np.array(qvel)

        e_pos = qpos[0:3] - self.desired_position               # position error

        if self.current_quat is not None:
            self.previous_quat = self.current_quat.copy()       # rotation matrix

        quat = np.array(qpos[3:7])
        self.current_quat = np.array(quat)
        rot_mat = utils.quat2rot(quat)                          # rotation matrix
        vel = qvel[0:3] # np.array(self.sim.data.get_joint_qvel("root")[:3])

        # angular_vel = np.array(self.sim.data.get_body_xvelr("core"))

        angular_vel = qvel[3:6] # np.array(self.sim.data.get_joint_qvel("root")[3:6])     # angular velocity of core of the robot in body frame.

        omegas = state[13:17]

        return np.concatenate([e_pos, quat, vel, angular_vel, omegas]).flatten()

    def get_reward(self, ob, a):
        """
        Method to evaluate reward based on observation and action

        Args:
            ob (numpy.ndarray): Observation vector.
            a (numpy.ndarray): Action vector of shape (4,).

        Returns:
            tuple[float, dict]: Tuple containing follwing elements in the given order:
                - reward (float): Scalar reward based on observation and action.
                - reward_info (dict): Dictionary of reward for specific state values. This dictionary contains the reward values corresponding to the following keys - (position, orientation, linear_velocity, angular_velocity, action, alive_bonus, extra_bonus, extra_penalty).
        """

        alive_bonus = self.reward_for_staying_alive

        reward_position = self.norm(ob[0:3]) * (-self.position_reward_constant)

        reward_orientation = self.orientation_error(ob[3:7]) * (-self.orientation_reward_constant)

        reward_linear_velocity = self.norm(ob[7:10]) * (-self.linear_velocity_reward_constant)

        reward_angular_velocity = self.norm(ob[10:13]) * (-self.angular_velocity_reward_constant)

        reward_action = self.norm(a) * (-self.action_reward_constant)

        extra_bonus = self.bonus_reward_to_achieve_goal(ob[:3])   # EXTRA BONUS TO ACHIEVE THE GOAL

        extra_penalty = - self.bound_violation_penalty(ob[:3])    # PENALTY for bound violation

        reward_velocity_towards_goal = 0.0
        if self.norm(ob[0:3]) > self.error_tolerance_norm:  # reward agent to move towards goal if system is away from goal
            reward_velocity_towards_goal += self.reward_velocity_towards_goal(error_xyz=ob[:3], velocity=ob[7:10])

        rewards = (reward_position, reward_orientation, reward_linear_velocity, reward_angular_velocity, reward_action, alive_bonus, extra_bonus, extra_penalty, reward_velocity_towards_goal)
        reward = sum(rewards) * self.reward_scaling_coefficient

        reward_info = dict(
            position=reward_position,
            orientation=reward_orientation,
            linear_velocity=reward_linear_velocity,
            angular_velocity=reward_angular_velocity,
            action=reward_action,
            alive_bonus=alive_bonus,
            extra_bonus=extra_bonus,
            extra_penalty=extra_penalty,
            velocity_towards_goal=reward_velocity_towards_goal,
            all=rewards
        )

        return reward, reward_info

    def _random_state(self, randomize=True):
        """
        Method to initial the robot in Simulation environment.

        Args:
            randomize (bool): If ``True``, initialize the robot randomly.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Tuple containing the following vectors in given order:
                - qpose_init (numpy.ndarray): Vector of robot's state after perturbation (dim-18).
                - qvel_init (numpy.ndarray): Vector of robot's velocity after perturbation (dim-6).
        """

        if not randomize:
            qpos_init = np.array([self.desired_position[0], self.desired_position[1], self.desired_position[2], 1., 0., 0., 0.])
            qvel_init = np.zeros((6,))
            return qpos_init, qvel_init

        # attitude (roll pitch yaw)
        quat_init = np.array([1., 0., 0., 0.])
        if self.disorient and self.sample_SO3:
            rot_mat = utils.sampleSO3()
            quat_init = utils.rot2quat(rot_mat)
        elif self.disorient:
            attitude_euler_rand = self.np_random.uniform(low=-self.init_max_attitude, high=self.init_max_attitude, size=(3,))
            quat_init = utils.euler2quat(attitude_euler_rand)

        # position (x, y, z)
        c = 0.2
        ep = self.np_random.uniform(low=-(self.env_bounding_box-c), high=(self.env_bounding_box-c), size=(3,))
        pos_init = ep + self.desired_position

        # velocity (vx, vy, vz)
        vel_init = utils.sample_unit3d() * self.init_max_vel

        # angular velocity (wx, wy, wz)
        angular_vel_init = utils.sample_unit3d() * self.init_max_angular_vel

        # set states
        qpos_init = np.concatenate([pos_init, quat_init]).ravel()
        qvel_init = np.concatenate([vel_init, angular_vel_init]).ravel()

        # omegas 
        omegas_init = np.array([self.quad.params['w_hover']]*4)

        state_init = np.concatenate([pos_init, quat_init, vel_init, angular_vel_init, omegas_init]).ravel()

        return state_init
    
    def _get_info(self):
        return {}
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)

        random_state = self._random_state()

        self.quad.set_state(random_state)

        self.quad.t = self.quad.Ti

        # self.cylinder_params = self._random_cylinder()

        self._time = 0

        obs = self._get_obs()

        info = self._get_info()

        return obs, info
    
    
    def close(self):
        if self.quad.render == 'online_mujoco':
            self.quad.viewer.close()

