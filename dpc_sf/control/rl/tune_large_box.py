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
TOTAL_TIMESTEPS = 1_000_000         # 10_000_000 for ppo, 2_000_000 for SAC
MODEL_TYPE = 'SAC'                  # 'PPO', 'SAC'
BACKEND = 'mj'                      # 'mj', 'eom'
MJ_RENDER = True
PRETRAIN_TYPE = 'small_box'            # 'small_box', 'large_box', 'obstacle'
TRAIN_TYPE = 'large_box'

# initial conditions
state = params["default_init_state_np"]
reference = waypoint_reference('wp_p2p', average_vel=1.0)

if TRAIN_TYPE == 'small_box':
    env_bounding_box = 1.2
elif TRAIN_TYPE == 'large_box':
    env_bounding_box = 5.0


register(
    id='QuadrotorXHoverEnv-v0',
    entry_point='dpc_sf.gym_environments.quadcopter_x_hover5_norm_obs:QuadrotorXHoverEnv',
    kwargs=dict(env_bounding_box=env_bounding_box, randomize_reset=True, state=state, reference=reference, backend=BACKEND),
)

# normalise the reward
vec_env = make_vec_env(ENV_NAME, n_envs=8, seed=SEED)
vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

# vec_env = VecNormalize.load(f"./policy/{MODEL_TYPE}_{BACKEND}_{ENV_NAME}/{PRETRAIN_TYPE}/vec_env.pkl", vec_env)
# prevent updates to statistical normalisation with this flag
# vec_env.training = False

if BACKEND == 'mj' and MJ_RENDER == True:
    vec_env.envs[0].quad.start_online_render()

if MODEL_TYPE == 'SAC':
    model = SAC.load(path=f"./policy/{MODEL_TYPE}_{BACKEND}_{ENV_NAME}/{PRETRAIN_TYPE}/model", env=vec_env)
elif MODEL_TYPE == 'PPO':
    model = PPO.load(path=f"./policy/{MODEL_TYPE}_{BACKEND}_{ENV_NAME}/{PRETRAIN_TYPE}/model", env=vec_env)

eval_callback = EvalCallback(vec_env, best_model_save_path=f"./policy/",
                            log_path='./logs/', eval_freq=int(10000/8),
                            deterministic=True, render=False)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

model.save(f"./policy/{MODEL_TYPE}_{BACKEND}_{ENV_NAME}/{TRAIN_TYPE}/model")

# runs the model
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    
del model

vec_env.save(f"./policy/{MODEL_TYPE}_{BACKEND}_{ENV_NAME}/{TRAIN_TYPE}/vec_env.pkl")
vec_env.close()

