import torch
import numpy as np


# Define replay buffer object
class ReplayBuffer:
    def __init__(self, N, obs_space_shape, action_space_shape):
        self.obs = torch.zeros((N + 1, *obs_space_shape), dtype=torch.float32)
        self.actions = torch.zeros((N, *action_space_shape), dtype=torch.float32)
        self.rewards = torch.zeros(N, dtype=torch.float32)
        self.done = torch.zeros(N, dtype=torch.bool)
        
        self.rtg = torch.zeros(N, dtype=torch.float32)
        
        self.N = N
        self.idx = 0

    
    def store(self, obs, action, reward, done):
        self.obs[self.idx] = torch.from_numpy(obs).float()
        self.actions[self.idx] = torch.from_numpy(action).float()
        self.rewards[self.idx] = reward
        self.done[self.idx] = done
        self.idx += 1

        if self.idx == self.N:
            self.idx = 0
            return True  # return value indicates timeout
        else:
            return False

    
    def store_terminal_obs(self, obs):
        self.obs[self.N] = torch.from_numpy(obs).float()

    
    def compute_rewards_to_go(self):
        for i in reversed(range(self.N)):
            

    def compute_gae(self):
        pass
