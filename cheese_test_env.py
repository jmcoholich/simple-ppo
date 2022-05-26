import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class PendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0):

        high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        rew = float(self.state[0] * u > 0)
        self._update_state()

        return self.state, rew, False, {}

    def reset(self):
        self._update_state()
        return self.state

    def _update_state(self):
        if np.random.choice([-1, 1]) == 1:
            self.state = np.ones(3)
        else:
            self.state = -np.ones(3)

    def render(self, mode="human"):
        return

    def close(self):
        return
