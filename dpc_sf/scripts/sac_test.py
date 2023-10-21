from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.envs import SimpleMultiObsEnv
import gymnasium as gym


# Stable Baselines provides SimpleMultiObsEnv as an example environment with Dict observations
env = gym.make("Pendulum-v1", render_mode="human")

n_sampled_goal = 4
model = SAC(
    "MultiInputPolicy",
    env,
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

model.learn(total_timesteps=100_000)