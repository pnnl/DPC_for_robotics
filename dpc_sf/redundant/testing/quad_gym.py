import gymnasium as gym
from gymnasium import spaces
import numpy as np

class QuadcopterGym(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, quad=None, render_mode=None) -> None:

        state_lower_bound = np.array(quad.raw_constraints(0))
        state_upper_bound = np.array(quad.raw_constraints(1))
        
        # observation space size is implied by the shape of the bounds
        self.observation_space = spaces.Dict({
                "agent": spaces.Box(state_lower_bound, state_upper_bound, dtype=float),
                "target": spaces.Box(state_lower_bound, state_upper_bound, dtype=float)
            })
        
        self.action_space = spaces.Box(-100, 100, dtype=float)

        


