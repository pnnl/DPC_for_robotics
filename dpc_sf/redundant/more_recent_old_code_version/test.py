import math
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

from stable_baselines3 import PPO
screen_width = 900

class GameEnv(Env):
    def __init__(self):
        self.action_space = Discrete(5)

        observation_positions = Box(low=0, high=screen_width, shape=(2,))

        self.observation_space = Dict({'observation_positions': observation_positions})

        self.state = self.observation_space.sample()

    def step(self, action):
        self.state = self.observation_space.sample()

    def render(self):
        pass

    def reset(self, seed):
        return self.state


env = GameEnv()

model = PPO('MultiInputPolicy', env, verbose=1,)
model.learn(total_timesteps=1000)