from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
from dpc_sf.control.trajectory.trajectory import waypoint_reference
from dpc_sf.dynamics.params import params
import torch

SEED = 123
ENV_NAME = "QuadrotorXHoverEnv-v0"
MODEL_TYPE = 'SAC'                  # 'PPO', 'SAC'
BACKEND = 'mj'                      # 'mj', 'eom'
MJ_RENDER = True
PRETRAIN_TYPE = 'small_box'            # 'small_box', 'large_box', 'obstacle'

# initial conditions
state = params["default_init_state_np"]
reference = waypoint_reference('wp_p2p', average_vel=1.0)

if PRETRAIN_TYPE == 'small_box':
    env_bounding_box = 1.2
elif PRETRAIN_TYPE == 'large_box':
    env_bounding_box = 5.0

register(
    id='QuadrotorXHoverEnv-v0',
    entry_point='dpc_sf.gym_environments.quadcopter_x_hover5:QuadrotorXHoverEnv',
    kwargs=dict(env_bounding_box=env_bounding_box, randomize_reset=True, state=state, reference=reference, backend=BACKEND),
)

vec_env = make_vec_env(ENV_NAME, n_envs=8, seed=SEED)
vec_env = VecNormalize.load(f"./policy/{MODEL_TYPE}_{BACKEND}_{ENV_NAME}/{PRETRAIN_TYPE}/vec_env.pkl", vec_env)
# prevent updates to statistical normalisation with this flag
vec_env.training = False
vec_env.norm_reward = False

if BACKEND == 'mj' and MJ_RENDER == True:
    vec_env.envs[0].quad.start_online_render()

if MODEL_TYPE == 'SAC':
    model = SAC.load(path=f"./policy/{MODEL_TYPE}_{BACKEND}_{ENV_NAME}/{PRETRAIN_TYPE}/model", env=vec_env)
elif MODEL_TYPE == 'PPO':
    model = PPO.load(path=f"./policy/{MODEL_TYPE}_{BACKEND}_{ENV_NAME}/{PRETRAIN_TYPE}/model", env=vec_env)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, info = vec_env.step(action)
    if terminated[0]:
        obs = vec_env.reset()

