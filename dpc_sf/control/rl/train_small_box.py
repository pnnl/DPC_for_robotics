from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
from dpc_sf.control.trajectory.trajectory import waypoint_reference
from dpc_sf.control.rl.callback import SaveVecNormalizeCallback
from dpc_sf.dynamics.params import params
import torch
import gymnasium as gym

SEED = 123
ENV_NAME = "QuadrotorXHoverEnv-v0"
TOTAL_TIMESTEPS = 5_000_000         # 10_000_000 for ppo, 1_000_000 for SAC
MODEL_TYPE = 'SAC'                  # 'PPO', 'SAC'
BACKEND = 'eom'                      # 'mj', 'eom'
MJ_RENDER = True
TRAIN_TYPE = 'small_box'

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

vec_env = make_vec_env(ENV_NAME, n_envs=8, seed=SEED)
# CHANGE NORM OBS TO TRUE FOR NON MANUALLY NORMALISED ENV!!!!
vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

if BACKEND == 'mj' and MJ_RENDER == True:
    vec_env.envs[0].quad.start_online_render()

if MODEL_TYPE == 'SAC':
    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=f'logs/log_{ENV_NAME}',
        policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256], qf=[256, 256])),
        learning_rate=0.00005,
        seed=SEED,
        batch_size=256,
    )
elif MODEL_TYPE == 'PPO':
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

eval_callback = EvalCallback(vec_env, best_model_save_path=f"./policy/{MODEL_TYPE}_{BACKEND}_{ENV_NAME}/{TRAIN_TYPE}/",
                            log_path='./logs/', eval_freq=int(10000/8),
                            deterministic=True, render=False)
norm_stats_callback = SaveVecNormalizeCallback(save_freq=int(10000/8), save_path=f"./policy/{MODEL_TYPE}_{BACKEND}_{ENV_NAME}/{TRAIN_TYPE}/", verbose=1)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_callback, norm_stats_callback])

model.save(f"./policy/{MODEL_TYPE}_{BACKEND}_{ENV_NAME}/{TRAIN_TYPE}/model")


# load_model = SAC.load(path=f"./policy/{MODEL_TYPE}_{BACKEND}_{ENV_NAME}/{TRAIN_TYPE}/model")
# 
# new_env = make_vec_env(ENV_NAME, n_envs=8, seed=SEED)
# new_env.envs[0].quad.start_online_render()
# 
# # runs the model
# obs = new_env.reset()
# for i in range(1000):
#     action, _states = load_model.predict(obs)
#     obs, rewards, dones, info = new_env.step(action)
# 
# 
# del model
# 
# vec_env.save(f"./policy/{MODEL_TYPE}_{BACKEND}_{ENV_NAME}/{TRAIN_TYPE}/vec_env.pkl")
# vec_env.close()
# 
# from stable_baselines3.common.evaluation import evaluate_policy
