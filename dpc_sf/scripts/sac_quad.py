# currently reinforcement learning is not a vibe on macos - waiting to get access to HPC
# I now have access to HPC
# testing GPU access with this script on the head node on marianas with nvidia access

import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.env_checker import check_env
import torch
from dpc_sf.gym_environments.dict_test import QuadcopterGymP2P
from dpc_sf.dynamics.params import params
from dpc_sf.control.trajectory.trajectory import waypoint_reference
import numpy as np
import dpc_sf.utils.pytorch_utils as ptu
import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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

env = QuadcopterGymP2P(
    state=state,
    reference=reference,
    Q = Q,
    R = R,
    Ts = 0.1,
    Ti = 0.0,
    Tf = 4.0,
    params = params,
    backend = 'mj',
    integrator = 'euler'
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
HalfCheetah-v4 spec:
EnvSpec(
    id='HalfCheetah-v4', entry_point='gymnasium.envs.mujoco.half_cheetah_v4:HalfCheetahEnv', 
    reward_threshold=4800.0, nondeterministic=False, max_episode_steps=1000, order_enforce=True, 
    autoreset=False, disable_env_checker=False, apply_api_compatibility=False, kwargs={}, 
    namespace=None, name='HalfCheetah', version=4, additional_wrappers=(), vector_entry_point=None
)
"""
gym.register(
    id='Quad-v0',
    entry_point='dpc_sf.gym_environments.dict_test:QuadcopterGymP2P',
    reward_threshold=1000,
    nondeterministic=False,
    max_episode_steps=300,
    order_enforce=True,
    autoreset=False,
    disable_env_checker=False,
    apply_api_compatibility=False,
    additional_wrappers=(),
    vector_entry_point=None,
    kwargs={
        'state': state,
        'reference': reference,
        'Q': Q,
        'R': R
    }
)

env = gym.make('Quad-v0')
check_env(env, warn=True)

vec_env = DummyVecEnv([lambda: gym.make("Quad-v0")])
# Automatically normalize the input features and reward
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)

# model = SAC("MlpPolicy", vec_env, verbose=1, device=device)
# model = SAC("MultiInputPolicy", vec_env, verbose=1, device=device)

# SAC hyperparams: (these were for the parking task: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)
n_sampled_goal = 4
model = SAC(
    "MultiInputPolicy",
    vec_env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
      n_sampled_goal=n_sampled_goal,
      goal_selection_strategy="future",
    ),
    verbose=1,
    buffer_size=int(1e6),
    learning_rate=1e-3,
    gamma=0.95,
    batch_size=256,
    policy_kwargs=dict(net_arch=[256, 256, 256]),
)

# model = SAC("MlpPolicy", env, verbose=1, device=device)
model.learn(total_timesteps=int(2e5), log_interval=4)

# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = "/tmp/"
model.save(log_dir + "ppo_halfcheetah")
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
env.save(stats_path)

# To demonstrate loading
del model, vec_env

# Load the saved statistics
vec_env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
vec_env = VecNormalize.load(stats_path, vec_env)
#  do not update them at test time
vec_env.training = False
# reward normalization is not needed at test time
vec_env.norm_reward = False

# Load the agent
model = SAC.load(log_dir + "ppo_halfcheetah", env=vec_env)


raise Exception('stop')

env.animate()
import matplotlib.pyplot as plt
plt.show()

del model # remove to demonstrate saving and loading

model = SAC.load("sac_quad")

obs, info = env.reset()
i = 0
while True:
    print(i)
    i += 1
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated,_, info = env.step(action)
    if terminated:
        obs, info = env.reset()

