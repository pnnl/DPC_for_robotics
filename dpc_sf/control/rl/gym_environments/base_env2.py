# *************************************************************************
# This file is a heavily modified version of the following code:
# https://github.com/ethz-asl/reinmav-gym/blob/master/gym_reinmav/envs/mujoco/mujoco_quad.py
# The corresponding copyright notice is provided below.
#
# Copyright (c) 2019, Autonomous Systems Lab
# Author: Dongho Kang <eastsky.kang@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# *************************************************************************


import os
import math
from abc import ABC
from typing import Tuple, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium import utils as gym_utils
from gymnasium import Env
# from gym_multirotor import utils as multirotor_utils
import torch
import copy

import dpc_sf
import dpc_sf.utils.pytorch_utils as ptu

from dpc_sf.dynamics.params import params
from dpc_sf.utils.mixer import mixerFM_np
from dpc_sf.dynamics.eom_pt import QuadcopterPT
from dpc_sf.dynamics.mj import QuadcopterMJ
from dpc_sf.control.rl.gym_environments import multirotor_utils


class UAVBaseEnv(Env, gym_utils.EzPickle, ABC):
    """Abstract base class for UAV environment.

    Args:
        xml_name (str): Name of the robot description xml file.
        frame_skip (int): Number of steps to skip after application of control command.
        error_tolerance (float): Error tolerance. Default is `0.05`.
        max_time_steps (int): Maximum number of timesteps in each episode. Default is `2500`.
        randomize_reset (bool): If `True`, initailize the environment with random state at the start of each episode. Default is `True`.
        disorient (bool): If True, random initialization and random orientation of the system at start of each episode. Default is ``True``.
        sample_SO3 (bool): If True, sample orientation uniformly from SO3 for random initialization of episode. Default is `False`.
        observation_noise_std (float): Standard deviation of noise added to observation vector. If zero, no noise is added to observation. Default is `0.`. If non-zero, a vector is sampled from normal distribution and added to observation vector.
        reduce_heading_error (bool): If `True`, reward function tries to reduce the heading error in orientation of the system. Default is `True`.
        env_bounding_box (float): Max. initial position error. Use to initialize the system in bounded space. Default is `1.2`.
        init_max_vel (float): Max. initial velocity error. Use to initialize the system in bounded space. Default is ``0.5``.
        init_max_angular_vel (float): Max. initial angular velocity error. Use to initialize the system in bounded space. Default is `pi/10`.
        init_max_attitude (float): Max. initial attitude error. Use to initialize the system in bounded space. Default is `pi/3.`.
        bonus_to_reach_goal (float): Bonus value or reward when RL agent takes the system to the goal state. Default is ``15.0``.
        max_reward_for_velocity_towards_goal (float): Max. reward possible when the agent is heading towards goal, i.e., velocity vector points towards goal direction. Default is ``2.0``.
        position_reward_constant (float): Position reward constant coefficient. Default is ``5.0``.
        orientation_reward_constant (float): Orientation reward constant coefficient. Default is ``0.02``.
        linear_velocity_reward_constant (float): Linear velocity reward constant. Default is ``0.01``.
        angular_velocity_reward_constant (float): Angular velocity reward constant. Default is ``0.001``.
        action_reward_constant (float): Action reward coefficient. Default is ``0.0025``.
        reward_for_staying_alive (float): Reward for staying alive in the environment. Default is ``5.0``.
        reward_scaling_coefficient (float): Reward multiplication factor which can be used to scale the value of reward to be greater or smaller. Default is ``1.0``.

    Notes:
    -  Need to implement ``reset_model`` in every subclass along with few other methods..

    """

    obs_xyz_index = np.arange(0, 3)
    obs_rot_mat_index = np.arange(3, 12)
    obs_vel_index = np.arange(12, 15)
    obs_avel_index = np.arange(15, 18)

    action_index_thrust = np.arange(0, 4)

    def __init__(
            self,

            # quad requirements
            state: torch.Tensor, # initial condition
            reference,
            params=params,
            Ts: float = 0.01,
            Ti: float = 0.0,
            Tf: float = 4.0,
            integrator='euler', # 'euler', 'rk4'
            # xml_path="quad_dynamics/quadrotor_x.xml", 
            # write_path="media/mujoco/",
            # render='matplotlib', # 'matplotlib', 'mujoco'
            backend='mj', # 'mj', 'eom'

            # gym requirements
            error_tolerance=0.05,
            max_time_steps=1000,
            randomize_reset=True,
            disorient=True,
            sample_SO3=False,
            observation_noise_std=0,
            reduce_heading_error=True,
            env_bounding_box=1.2, # was 1.2
            init_max_vel=0.5,
            init_max_angular_vel=0.1*math.pi,
            init_max_attitude=math.pi/3.0,
            bonus_to_reach_goal=15.0,
            max_reward_for_velocity_towards_goal=2.0,
            position_reward_constant=5.0,
            orientation_reward_constant=0.02,
            linear_velocity_reward_constant=0.01,
            angular_velocity_reward_constant=0.001,
            action_reward_constant=0.0025,
            reward_for_staying_alive=5.0,
            reward_scaling_coefficient=1.0
        ):

        if backend == 'eom':
            self.quad = QuadcopterPT(
                state=ptu.from_numpy(state),
                reference=reference,
                params=params,
                Ts=Ts,
                Ti=Ti,
                Tf=Tf,
                integrator=integrator
            )

        elif backend == 'mj':
            self.quad = QuadcopterMJ(
                state=state,
                reference=reference,
                params=params,
                Ts=Ts,
                Ti=Ti,
                Tf=Tf,
                integrator=integrator,
                xml_path="quadrotor_x.xml",
                write_path="media/mujoco/",
                render='matplotlib'
            )

        else:
            raise Exception("invalid backend choice")

        self.error_tolerance = error_tolerance
        """float: Error tolerance. Default is `0.05`."""

        self.max_time_steps = max_time_steps
        """
        int: Maximum number of timesteps in each episode. Default is `2500`.
        """

        # episode initialization parameters
        self.randomize_reset = randomize_reset
        """bool: If `True`, initailize the environment with random state at the start of each episode. Default is `True`.
        """

        self.disorient = disorient
        """bool: If True, random initialization and random orientation of the system at start of each episode. Default is ``True``.

        Notes:
            * If `self.disorient` is true and `self.sample_SO3` is true, randomly initialize orientation of robot for every episode and sample this orientation from uniform distribution of SO3 matrices.
            * If `self.disorient` is true, then randomly initialize the robot orientation at episode start.
            * If `self.disorient` is false, do not randomly initialize initial orientation of robot and do a deterministic
            initialization of the robot orientation as quaternion [1.0, 0., 0., 0.] for every episode.
        """

        self.sample_SO3 = sample_SO3
        """bool: If True, sample orientation uniformly from SO3 for random initialization of episode. Default is `False`.
        """

        self.observation_noise_std = observation_noise_std
        """float: Standard deviation of noise added to observation vector. If zero, no noise is added to observation. Default is `0.`. If non-zero, a vector is sampled from normal distribution and added to observation vector.
        """

        self.reduce_heading_error = reduce_heading_error
        """bool: If `True`, reward function tries to reduce the heading error in orientation of the system. Default is `True`."""

        # initial state randomizer bounds
        self.env_bounding_box = env_bounding_box
        """float: Max. initial position error. Use to initialize the system in bounded space. Default is ``1.2``."""

        self.init_max_vel = init_max_vel
        """float: Max. initial velocity error. Use to initialize the system in bounded space. Default is ``0.5``."""

        self.init_max_angular_vel = init_max_angular_vel
        """float: Max. initial angular velocity error. Use to initialize the system in bounded space. Default is `pi/10`."""

        self.init_max_attitude = init_max_attitude
        """float: Max. initial attitude error. Use to initialize the system in bounded space. Default is `pi/6.`."""

        self.bonus_to_reach_goal = bonus_to_reach_goal
        """float: Bonus value or reward when RL agent takes the system to the goal state. Default is ``15.0``.
        """

        self.max_reward_for_velocity_towards_goal = max_reward_for_velocity_towards_goal
        """float: Max. reward possible when the agent is heading towards goal, i.e., velocity vector points towards goal direction. Default is ``2.0``."""

        self.position_reward_constant = position_reward_constant
        """float: Position reward constant coefficient. Default is ``5.0``.
        """

        self.orientation_reward_constant = orientation_reward_constant
        """float: Orientation reward constant coefficient. Default is ``0.02``.
        """

        self.linear_velocity_reward_constant = linear_velocity_reward_constant
        """float: Linear velocity reward constant. Default is ``0.01``.
        """

        self.angular_velocity_reward_constant = angular_velocity_reward_constant
        """float: Angular velocity reward constant. Default is ``0.001``.
        """

        self.action_reward_constant = action_reward_constant
        """float: Action reward coefficient. Default is ``0.0025``.
        """

        self.reward_for_staying_alive = reward_for_staying_alive
        """float: Reward for staying alive in the environment. Default is ``5.0``.
        """

        self.reward_scaling_coefficient = reward_scaling_coefficient
        """float: Reward multiplication factor which can be used to scale the value of reward to be greater or smaller. Default is ``1.0``.
        """

        self.policy_range = [-1.0, 1.0]
        """tuple: Policy-output (a.k.a. action) range. Default is ``[-1.0, 1.0]``
        """

        self.policy_range_safe = [-0.8, 0.8]
        """tuple: Safe policy output range. Default is ``[-0.8, 0.8]``
        """

        # to record true states of robot in the simulator
        self.mujoco_qpos = None
        """numpy.ndarray: Mujoco pose vector of the system.
        """

        self.mujoco_qvel = None
        """numpy.ndarray: Mujoco velocity vector of the system.
        """

        self.previous_robot_observation = None
        """numpy.ndarray: Observation buffer to store previous robot observation
        """

        self.current_robot_observation = None
        """numpy.ndarray: Observation buffer to store current robot observation
        """

        self.previous_quat = None
        """numpy.ndarray: Buffer to keep record of orientation from mujoco. Stores previous quaternion.
        """

        self.current_quat = None
        """numpy.ndarray: Buffer to keep record of orientation from mujoco. Stores current orientation quaternion.
        """

        self.current_policy_action = np.array([-1, -1, -1, -1.])
        """numpy.ndarray: Buffer to hold current action input from policy to the environment. Stores the action vector scaled between (-1., 1.)
        """

        self.desired_position = np.array([0, 0, 3.0])
        """numpy.ndarray: Desired position of the system. Goal of the RL agent."""

        self._time = 0              # initialize time counter.
        self.gravity_mag = 9.81     # default value of acceleration due to gravity

        gym_utils.EzPickle.__init__(self)
        # mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip)

        self.gravity_mag = self.quad.params['g']



    @property
    def env_bounding_box_norm(self) -> float:
        """Max. distance of the drone from the bounding box limits or maximum allowed distance
        of the drone from the desired position. It is the radius of the sphere within which the robot can observe the goal.

        Returns:
            float: env_bounding_box_sphere_radius

        """
        return self.norm(np.array([self.env_bounding_box, self.env_bounding_box, self.env_bounding_box]))

    @property
    def error_tolerance_norm(self) -> float:
        """Returns the radius of the sphere within which the robot can be considered accurate.

        Returns:
            float: error_tolerance_norm_sphere_radius

        """
        return self.norm(np.array([self.error_tolerance, self.error_tolerance, self.error_tolerance]))

#     def step(self, a: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
#         """Take a step in the environment given an action
# 
#         Args:
#             a (numpy.ndarray): action from control policy
# 
#         Returns:
#             Tuple[np.ndarray, float, bool, Dict]: tuple of agent-next_state, agent-reward, episode completetion flag and additional-info
# 
#         """
#         reward = 0
#         self.do_simulation(self.clip_action(a), self.frame_skip)
#         ob = self._get_obs()
#         notdone = np.isfinite(ob).all()
#         done = not notdone
#         info = {"reward_info": reward, "mujoco_qpos": self.mujoco_qpos, "mujoco_qvel": self.mujoco_qvel}
#         return ob, reward, done, info

    def step(self, cmd: np.ndarray):

        # cmd is a desired set thrusts and torques
        cmd_w = mixerFM_np(self.quad.params, cmd[0], cmd[1:])
        w_error = cmd_w - self.quad.state.squeeze()[13:]

        # gives the exact desired omegas for the next iteration when using EoM
        p_gain = self.quad.params["IRzz"] / self.quad.Ts

        ctrl = w_error * p_gain

        self.quad.step(ctrl)

        obs = self._get_obs()

        reward = -self._cost(ctrl)

        done = self._is_terminal()

        info = self._get_info()

        return obs, reward, done, False, info

    def clip_action(self, action: np.ndarray, a_min=-1.0, a_max=1.0) -> np.ndarray:
        """Clip policy action vector to be within given minimum and maximum limits.

        Args:
            action (np.ndarray): Action vector
            a_min (float, optional): Lower bound for the action. Defaults to -1.0.
            a_max (float, optional): Upper bound for the action. Defaults to 1.0.

        Returns:
            np.ndarray: action vector bounded between lower and upper bound of action taken using the control policy.

        """
        action = np.clip(action, a_min=a_min, a_max=a_max)
        return action

    def viewer_setup(self):
        """This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position and so forth.
        """
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 2.5

    @property
    def mass(self) -> float:
        """Mass of the environment-body or robot.

        Returns:
            float: mass of the robot

        """
        return 1.2

    @property
    def inertia(self) -> np.ndarray:
        """

        Returns:
            numpy.ndarray: Inertia matrix of the system.
        """
        return np.diag([0.0123, 0.0123, 0.0224])

    def get_motor_input(self, action):
        raise NotImplementedError

    def orientation_error(self, quat: np.ndarray) -> float:
        """Orientation error assuming desired orientation is (roll, pitch, yaw) = (0, 0, 0).

        Args:
            quat (numpy.ndarray): Orientation quaternion (q0, qx, qy, qz) in scalar first format of the robot (shaped (4,))

        Returns:
            float: Magnitude of error in orientation.

        """
        error = 0.
        rpy = multirotor_utils.quat2euler(quat)
        # rpy = np.flip(dpc_sf.utils.quatToYPR_ZYX_np(quat))
        if self.reduce_heading_error:
            error += self.norm(rpy)
        else:
            error += self.norm(rpy[:2])     # exclude error in yaw

        return error

    def goal_reached(self, error_xyz: np.ndarray) -> bool:
        """This method checks if the given position error vector is close to zero or not.

        Args:
            error_xyz (numpy.ndarray): Vector of error along xyz axes.

        Returns:
            bool: ``True`` if the system reached the goal position else ``False``.

        Notes:
            - If the system reaches the goal position, error_xyz vector will be close to zero in magnitude.

        """
        return self.norm(error_xyz) < self.error_tolerance_norm

    def is_within_env_bounds(self, error_xyz: np.ndarray) -> bool:
        """This method checks if the robot is with the environment bounds or not.

        Args:
            error_xyz (numpy.ndarray): Vector of position error of the robot w.r.t. the target locations, i.e., vector = (target - robot_xyz).

        Returns:
            bool: ``True`` if the drone is within the environment limits else ``False``.

        Notes:
            - Environment bounds signify the range withing which the goal is observable.
            - If this function returns ``True``, it automatically means that the goal is within the observable range of robot.

        """
        return self.norm(error_xyz) < self.env_bounding_box_norm

    def norm(self, vector: np.ndarray) -> float:
        """Helper calculate the euclidean norm of a vector.

        Args:
            vector (numpy.ndarray): Vector of shape (n,)

        Returns:
            float: Norm or magnitude of the vector.
        """
        return np.linalg.norm(vector)

    def bound_violation_penalty(self, error_xyz: np.ndarray) -> float:
        """

        Args:
            error_xyz (numpy.ndarray): Error vector of robot position and desired position along x-y-z axes.

        Returns:
            float: If the robot is within the goal range, than the penalty is zero, else the penalty is greater than zero.

        Notes:
            - Subtract penalty from reward or when using this value in reward function multiply it with ``-1``.

        """
        penalty = 0.0
        if not self.is_within_env_bounds(error_xyz):
            penalty += self.bonus_to_reach_goal
        return penalty

    def bonus_reward_to_achieve_goal(self, error_xyz: np.ndarray) -> float:
        """Return bonus reward value if the goal is achieved by the robot.

        Args:
            error_xyz (numpy.ndarray): Error vector of robot position and desired position along x-y-z axes.

        Returns:
            float: Bonus reward value if the goal is achieved using the robot and the control agent.

        """

        bonus = 0.0
        if self.goal_reached(error_xyz):
            bonus += self.bonus_to_reach_goal
        return bonus

    def reward_velocity_towards_goal(self, error_xyz: np.ndarray, velocity: np.ndarray) -> float:
        """Reward for velocity of the system towards goal.

        Args:
            error_xyz (numpy.ndarray): Position error of the system along xyz-coordinates.
            velocity (numpy.ndarray): Velocity vector (vx, vy, vz) of the system in body reference frame

        Returns:
            float: Reward based on the system velocity inline with the desired position.

        """
        if self.goal_reached(error_xyz):
            return self.max_reward_for_velocity_towards_goal
        unit_xyz = error_xyz/(self.norm(error_xyz) + 1e-6)
        velocity_direction = velocity/(self.norm(velocity) + 1e-6)
        reward = np.dot(unit_xyz, velocity_direction)
        return np.clip(reward, -np.inf, self.max_reward_for_velocity_towards_goal)

    def is_done(self, ob: np.ndarray) -> bool:
        """Check if episode is over.

        Args:
            ob (numpy.ndarray): Observation vector

        Returns:
            bool: ``True`` if current episode is over else ``False``.

        """
        notdone = np.isfinite(ob).all() and self.is_within_env_bounds(ob[:3]) and (self._time < self.max_time_steps)
        done = not notdone
        # if done:
        #     print('f')
        return done

#     def get_body_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         """Returns the state of the body with respect to world frame of reference (inertial frame).
# 
#         Args:
#             body_name (str): Name of the body used in the XML file.
# 
#         Returns:
#             Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: Tuple of state of body containing xyz ``(3,)``, rotation_matrix_flat ``(9,)``, linear_vel ``(3,)``, angular_vel ``(3,)``.
# 
#         """
#         state = self.quad.get_state()
# 
#         xyz = state[0:3]
#         quat = state[3:7]
#         vel = state[7:10]
#         avel = state[10:13]
# 
#         return xyz, quat, vel, avel
# 
#     def get_body_states_for_plots(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         """Returns the state of given body.
# 
#         Args:
#             body_name (str): Name of the body in mujoco xml
# 
#         Returns:
#             Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple of state of body containing xyz ``(3,)``, euler angles roll-pitch-yaw ``(3,)``, linear_vel ``(3,)``, angular_vel ``(3,)``.
#         """
#         xyz, quat, vel, avel = self.get_body_state()
#         rpy = np.flip(dpc_sf.utils.quatToYPR_ZYX_np(quat))
#         return xyz, rpy, vel, avel
# 
#     def get_body_state_in_body_frame(self, body_name: str, xyz_ref: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         """
# 
#         Args:
#             body_name (str): Name of the body in XML File
#             xyz_ref (numpy.ndarray): Reference XYZ of body frame
# 
#         Returns:
#             Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: xyz, mat, vel (in body frame of reference), avel
# 
#         """
#         xyz, quat, vel, avel = self.get_body_state(body_name)        # in world frame
# 
#         if xyz_ref is not None:
#             xyz = np.array(xyz) - np.array(xyz_ref)
# 
#         mat_ = mat.reshape((3, 3))
#         vel = np.dot(mat_, vel)  # vel in body frame
# 
#         return xyz, mat, vel, avel
