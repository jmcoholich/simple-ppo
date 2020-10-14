import torch
import numpy as np


# Define replay buffer object
class ReplayBuffer:
    def __init__(self, N, obs_space_shape, action_space_shape):
        '''initialize all tensors of information to be stored'''

        # Sample data
        self.obs = torch.zeros((N + 1, *obs_space_shape), dtype=torch.float32)
        self.actions = torch.zeros((N + 1, *action_space_shape), dtype=torch.float32)
        self.rewards = torch.zeros(N + 1, dtype=torch.float32)
        self.done = torch.zeros(N + 1, dtype=torch.bool)
        self.timeout = torch.zeros(N + 1, dtype=torch.bool)
        self.real_rewards = torch.zeros(N + 1, dtype=torch.float32)

        # Calculated/passed values
        self.rtg = torch.zeros(N, dtype=torch.float32)
        self.empirical_values = torch.zeros(N, dtype=torch.float32)
        self.advantages = torch.zeros(N, dtype=torch.float32)
        # self.predicted_values = torch.zeros(N + 1, dtype=torch.float32)

        self.N = N
        self.idx = 0


    def store(self, obs, action, reward, done, info):
        self.obs[self.idx] = torch.from_numpy(obs).float()
        self.actions[self.idx] = torch.from_numpy(action).float()
        self.rewards[self.idx] = reward
        self.done[self.idx] = done
        self.real_rewards[self.idx] = info['real_reward']
        if done:
            self.timeout[self.idx] = info['TimeLimit.truncated']

        if self.idx == self.N - 1: # if we just stored the second-to-last step
            # for later calculations, I say I did NOT timeout if I happened to reach a terminal state at the same time
            self.timeout[self.idx] = False if self.done[self.idx] else True 
            self.done[self.idx] = True
            self.idx += 1
            return True  # return value indicates buffer timeout
        elif self.idx == self.N: # if we just filled in the final "bootstrapped" step
            self.idx = 0
            # assert that timeout is never true when done is false
            assert self.done(torch.nonzero(self.timeout)).all()
            return False
        else:
            self.idx += 1
            return False


    def bootstrap_reward(self, obs, rew):
        '''Store the final state and get a bootstrapped final reward.'''
        # assert not self.done[-1]
        # assert len(self.obs) == self.N + 1
        self.obs[self.N] = torch.from_numpy(obs).float()
        self.rewards[-1] = rew


    # def compute_rewards_to_go(self):
    #     '''Computes rewards-to-go'''
    #     for i in reversed(range(self.N)):
    #         self.rtg[i] = self.rewards[i] + self.done[i] * (self.rtg[i + 1] if i < self.N - 1 else self.rewards[i])
    

    # def compute_empirical_values(self, gamma):
    #     '''Computes empirical values of each state, based on sample data (not from value function). However, this is
    #     using the bootstrapped final value from the value function.'''
    #     for i in reversed(range(self.N)):
    #         self.empirical_values[i] = self.rewards[i] + gamma * self.done[i] * (self.empirical_values[i + 1] if\
    #                                     i < self.N - 1 else self.rewards[i])


    def compute_gae(self, predicted_values, gamma, gae_lambda):
        ''' Computes generalized advantages estimates'''
        assert not (predicted_values == 0).all() # make sure that predicted values have actually been passed
        deltas = self.rewards[:-1] + gamma * predicted_values[1:] * (~self.done) - predicted_values[:-1]
        assert len(deltas) == self.N
        deltas = torch.cat((deltas, torch.Tensor([0.])))
        for i in reversed(range(self.N)):
            self.advantages[i] = deltas[i] + gamma * gae_lambda * deltas[i + 1] * (not self.done[i])

        # self.advantages -= self.advantages.mean()
        # self.advantages /= self.advantages.std() + 1e-10
