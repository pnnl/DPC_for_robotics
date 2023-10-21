
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import torch
from gymnasium.envs.registration import register
import gymnasium as gym

import numpy as np
from dpc_sf.control.trajectory.trajectory import waypoint_reference
from dpc_sf.dynamics.params import params

# initial conditions
state = params["default_init_state_np"]
reference = waypoint_reference('wp_p2p', average_vel=1.0)

register(
    id='QuadrotorXHoverEnv-v0',
    entry_point='dpc_sf.gym_environments.quadcopter_x_hover5:QuadrotorXHoverEnv',
    kwargs=dict(env_bounding_box=1.2, randomize_reset=True, state=state, reference=reference),
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
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=f'logs/log_{ENV_NAME}',
        policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256])),
        learning_rate=0.00005,
        clip_range=0.05,
        seed=SEED,
        batch_size=256,
        max_grad_norm=0.2
    )
    model.learn(total_timesteps=40_000_000)
    model.save(f"./policy/PPO_{ENV_NAME}_small_box")
    del model
    vec_env.close()

