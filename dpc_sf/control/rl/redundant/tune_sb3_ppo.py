
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import torch
from gymnasium.envs.registration import register
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback

import numpy as np
from dpc_sf.control.trajectory.trajectory import waypoint_reference

# initial conditions
state=np.array([[
    0,                  # x
    0,                  # y
    0,                  # z
    1,                  # q0
    0,                  # q1
    0,                  # q2
    0,                  # q3
    0,                  # xdot
    0,                  # ydot
    0,                  # zdot
    0,                  # p
    0,                  # q
    0,                  # r
    522.9847140714692,  # wM1
    522.9847140714692,  # wM2
    522.9847140714692,  # wM3
    522.9847140714692   # wM4
]])

reference = waypoint_reference('wp_p2p', average_vel=1.0)

Q = np.eye(17)
Q[13,13], Q[14,14], Q[15,15], Q[16,16] = 0, 0, 0, 0
R = np.eye(4)

register(
    id='QuadrotorXHoverEnv-v0',
    entry_point='dpc_sf.gym_environments.quadcopter_x_hover4:QuadrotorXHoverEnv',
    kwargs=dict(env_bounding_box=1.2, randomize_reset=True, state=state, reference=reference, Q=Q, R=R),
)

# env = gym.make('QuadrotorXHoverEnv-v0')
# check_env(env, warn=True)

SEED = 123
ENV_NAMES = ["QuadrotorXHoverEnv-v0"]#, "TiltrotorPlus8DofHoverEnv-v0", "QuadrotorPlusHoverEnv-v0"]
from stable_baselines3.common.vec_env import VecNormalize

for ENV_NAME in ENV_NAMES:
    # vec_env = gym.make(ENV_NAME) # make_vec_env(ENV_NAME, n_envs=8, seed=SEED)     # Parallel environments
    vec_env = make_vec_env(ENV_NAME, n_envs=8, seed=SEED)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    # vec_env.envs[0].quad.start_online_render()
    model = PPO.load(
        f"./policy/40k_PPO_QuadrotorXHoverEnv-v0",
        env=vec_env,
    )
    model.learn(total_timesteps=5_000_000)
    model.save(f"./policy/PPO_{ENV_NAME}")
    del model
    vec_env.close()

