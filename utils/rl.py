import numpy as np
import torch

import utils.pytorch as ptu
import utils.rotation as rotation

from utils.quad import applyMixerFM
from abc import ABC
from gymnasium import utils as gym_utils
from gymnasium import Env
from dynamics import get_quad_params
import copy


class rl_quad_wrapper:
    def __init__(self, backend='mujoco') -> None:
        self.backend = backend
        self.params = get_quad_params()
        self.state = self.params["default_init_state_np"]

    def step(self, ctrl):
        pass
    

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
            params=get_quad_params(),
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
            init_max_angular_vel=0.1*np.pi,
            init_max_attitude=np.pi/3.0,
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

        self.params = params

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

        self.gravity_mag = self.params['g']



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

    def step(self, cmd: np.ndarray):

        # cmd is a desired set thrusts and torques
        cmd_w = applyMixerFM.numpy(self.params, cmd[0], cmd[1:])
        w_error = cmd_w - self.quad.state.squeeze()[13:]

        # gives the exact desired omegas for the next iteration when using EoM
        p_gain = self.params["IRzz"] / self.quad.Ts

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
        rpy = rotation.quaternion_to_euler.numpy(quat)
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
            env_bounding_box=1.2, 
            randomize_reset=False,
            backend = 'mj',
        ):
        super().__init__(
            state=state,
            reference=reference,
            env_bounding_box=env_bounding_box, 
            randomize_reset=randomize_reset,
            backend=backend
        )

        self.init_state = state

        sa_dtype = np.float64

        self.action_space = spaces.Box(low=np.array([-1,-1,-1,-1]), high=np.array([1,1,1,1], dtype=sa_dtype))

        # post normalisation observation space
        self.observation_space = spaces.Box(
            low = np.array([-10]*20),
            high = np.array([10]*20),
            dtype=sa_dtype
        )

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
        state = self.quad.get_state()
        reference = self.quad.reference(self.quad.t)
        e_pos = state[0:3] - reference[0:3]

        if self.current_quat is not None:
            self.previous_quat = self.current_quat.copy() 

        quat = state[3:7]
        self.current_quat = quat
        cyl = np.array([1,1,0.5])

        raw_obs = np.concatenate([e_pos, state[3:], cyl]).flatten()

        norm_obs = normalize_rl(raw_obs)

        return norm_obs
    
    def is_within_env_bounds(self) -> bool:
        def point_segment_distance(point, line_start, line_end):
            line_vec = line_end - line_start
            point_vec = point - line_start

            # Calculate the t parameter for the point on the line segment
            t = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)

            # Check if the projection falls on the line segment
            if 0.0 <= t <= 1.0:
                proj = line_start + t * line_vec
                distance = np.linalg.norm(proj - point)
            else:
                # Calculate the distance to the closer end point
                distance = min(np.linalg.norm(point - line_start),
                            np.linalg.norm(point - line_end))
                
            return distance
        

        point = self.quad.get_state()[0:3]
        line_start = self.init_state[0:3]
        line_end = self.quad.reference(self.quad.t)[0:3]
        dist_from_tube = point_segment_distance(point, line_start, line_end)

        if dist_from_tube <= self.env_bounding_box_norm:
            is_within_env_bounds = True
        else:
            is_within_env_bounds = False
        return is_within_env_bounds
    
    def is_done(self, ob):
        """Check if episode is over.

        Args:
            ob (numpy.ndarray): Observation vector

        Returns:
            bool: ``True`` if current episode is over else ``False``.

        """

        notdone = np.isfinite(ob).all() and self.is_within_env_bounds() and (self._time < self.max_time_steps)
        done = not notdone
        # if done:
        #     print('f')
        return done
    
    def bound_violation_penalty(self, error_xyz) -> float:
        penalty = 0.0
        if not self.is_within_env_bounds():
            penalty += self.bonus_to_reach_goal
        return penalty

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
        desired_position = self.quad.reference(self.quad.t)[0:3]
        if not randomize:
            qpos_init = np.array([desired_position[0], desired_position[1], desired_position[2], 1., 0., 0., 0.])
            qvel_init = np.zeros((6,))
            return qpos_init, qvel_init

        # attitude (roll pitch yaw)
        quat_init = np.array([1., 0., 0., 0.])
        if self.disorient and self.sample_SO3:
            rot_mat = utils.sampleSO3()
            quat_init = rotation.rot_matrix_to_quaternion.numpy(rot_mat)
        elif self.disorient:
            attitude_euler_rand = self.np_random.uniform(low=-self.init_max_attitude, high=self.init_max_attitude, size=(3,))
            quat_init = rotation.euler_to_quaternion.numpy(attitude_euler_rand)

        # position (x, y, z)
        c = 0.2
        ep = self.np_random.uniform(low=-(self.env_bounding_box-c), high=(self.env_bounding_box-c), size=(3,))
        pos_init = ep + desired_position


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

        # for constructing the acceptable tube at the start of an episode
        self.init_state = copy.deepcopy(random_state)

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

