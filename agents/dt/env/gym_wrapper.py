"""Gymnasium wrapper around the Berkeley Pacman game engine."""
import gymnasium as gym
import numpy as np

class PacmanGymEnv(gym.Env):
    def __init__(self, layout_name='mediumClassic', render_mode=None):
        super().__init__()
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(256,), dtype=np.float32)
    def reset(self, **kwargs): raise NotImplementedError
    def step(self, action): raise NotImplementedError
    def render(self): raise NotImplementedError
