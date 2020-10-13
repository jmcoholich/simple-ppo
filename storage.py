import torch
import numpy as np


# Define replay buffer object
class ReplayBuffer:
    def __init__(self, N, obs_space_shape, action_space_shape):
        '''initialize all tensors of information to be stored'''

        # Sample data
        self.obs = torch.zeros((N + 1, *obs_space_shape), dtype=torch.float32)
        self.actions = torch.zeros((N, *action_space_shape), dtype=torch.float32)
        self.rewards = torch.zeros(N + 1, dtype=torch.float32)
        self.done = torch.zeros(N, dtype=torch.bool)
        self.real_rewards = torch.zeros(N, dtype=torch.float32)

        # Calculated values
        self.rtg = torch.zeros(N, dtype=torch.float32)
        self.empirical_values = torch.zeros(N, dtype=torch.float32)
        self.advantages = torch.zeros(N, dtype=torch.float32)

        self.N = N
        self.idx = 0


    def store(self, obs, action, reward, done, info):
        self.obs[self.idx] = torch.from_numpy(obs).float()
        self.actions[self.idx] = torch.from_numpy(action).float()
        self.rewards[self.idx] = reward
        self.done[self.idx] = done
        self.real_rewards[self.idx] = info['real_reward']
        self.idx += 1

        if self.idx == self.N:
            self.idx = 0
            return True  # return value indicates timeout
        else:
            return False


    def bootstrap_reward(self, obs, rew):
        '''Store the final state and get a bootstrapped final reward. (This method is only called if the final state
        is not terminal. '''
        assert not self.done[-1]
        assert len(self.obs) == self.N + 1
        self.obs[self.N] = torch.from_numpy(obs).float()
        self.rewards[-1] = rew


    def compute_rewards_to_go(self):
        '''Computes rewards-to-go'''
        for i in reversed(range(self.N)):
            self.rtg[i] = self.rewards[i] + self.done[i] * (self.rtg[i + 1] if i < self.N - 1 else self.rewards[i])
    

    def compute_empirical_values(self, gamma):
        '''Computes empirical values of each state, based on sample data (not from value function). However, this is
        using the bootstrapped final value from the value function.'''
        for i in reversed(range(self.N)):
            self.empirical_values[i] = self.rewards[i] + gamma * self.done[i] * (self.empirical_values[i + 1] if\
                                        i < self.N - 1 else self.rewards[i])


    def compute_gae(self, values, gamma, gae_lambda):
        ''' Computes generalized advantages estimates, AND afterwards standardizes all advantages'''
        values = values.squeeze()
        deltas = self.rewards[:-1] + gamma * values[1:] - values[:-1]
        assert len(deltas) == self.N
        deltas = torch.cat((deltas, torch.Tensor([0.])))
        for i in reversed(range(self.N)):
            self.advantages[i] = deltas[i] + gamma * gae_lambda * deltas[i + 1]

        self.advantages -= self.advantages.mean()
        self.advantages /= self.advantages.std() + 1e-10
