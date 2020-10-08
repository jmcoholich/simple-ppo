import gym 
import numpy as np
import torch
from torch import nn
from arguments import get_args
import os

# TODO 

# Define replay buffer object
class ReplayBuffer:
    def __init__(self, N, obs_space_shape, action_space_shape):
        self.obs = torch.zeros((N + 1, *obs_space_shape), dtype=torch.float32)
        self.actions = torch.zeros((N, *action_space_shape), dtype=torch.float32)
        self.rewards = torch.zeros(N, dtype=torch.float32)
        self.done = torch.zeros(N, dtype=torch.bool)
        self.N = N

        self.idx = 0


    def store(self, obs, action, reward, done):
        self.obs[self.idx] =  torch.from_numpy(obs).float()
        self.actions[self.idx] = torch.from_numpy(action).float()
        self.rewards[self.idx] = reward
        self.done[self.idx] = done
        self.idx += 1

        if self.idx == self.N:
            self.idx = 0
            return True # return value indicates timeout
        else:
            return False


    def store_terminal_obs(self, obs):
        self.obs[self.N] = torch.from_numpy(obs).float()


class Policy(nn.Module):
    def __init__(self, action_dim, obs_dim, hidden_size=256, hidden_layers=2, activation='relu', deterministic=False,
                independent_std=False):
        super(Policy, self).__init__() 

        action_dim, = action_dim
        obs_dim, = obs_dim
        activation = activation.lower()
        self.deterministic = deterministic
        self.independent_std = independent_std

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError('\'%s\' is not a currently supported activation function' % activation)

        self.linear_layers = [nn.Linear(obs_dim, hidden_size)]
        for _ in range(hidden_layers - 1):
            self.linear_layers.append(nn.Linear(hidden_size, hidden_size))
        self.linear_layers.append(nn.Linear(hidden_size, action_dim))

        if not deterministic: 
            if independent_std:
                log_std = -0.5 * torch.ones(action_dim).float()
                self.log_std = nn.Parameter(log_std, requires_grad=True)
            else:
                self.linear_log_std_layer = nn.Linear(hidden_size, action_dim)


    def forward(self, obs):
        ''' '''

        obs = torch.from_numpy(obs).float()
        for i in range(len(self.linear_layers) -1):
            obs = self.act(self.linear_layers[i](obs))

        if self.deterministic or self.independent_std: 
        # Just output the action means, no need to calculate standard deviations
            return self.linear_layers[-1](obs)
        else:
            # calculate and return means, log stds
            return self.linear_layers[-1](obs), self.linear_log_std_layer(obs)


    def get_action(self, obs):
        if self.deterministic:
            return self(obs)
        elif self.independent_std:
            mean = self(obs) 
            dist = torch.distributions.multivariate_normal.MultivariateNormal(mean,
                covariance_matrix=self.log_std.exp().diag())
            return dist.sample()
        else: 
            mean, log_std = self(obs)
            dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, 
                covariance_matrix=log_std.exp().diag())
            return dist.sample().numpy()


    def update(self):
        pass



class PPO:
    def __init__(self):
        pass


def main():
    args = get_args()

    num_updates = -(-args.num_env_steps//args.num_steps) # ceil division to ensure total steps are met

    env = gym.make('Pendulum-v0')
    memory = ReplayBuffer(args.num_steps, env.observation_space.high.shape, env.action_space.high.shape)
    policy = Policy(hidden_size=args.hidden_size,
                    hidden_layers=args.hidden_layers, 
                    activation=args.activation,
                    action_dim=env.action_space.high.shape,
                    obs_dim=env.observation_space.high.shape,
                    deterministic=args.deterministic,
                    independent_std=args.independent_std)


    for i in range(num_updates):
        obs = env.reset()
        
        while True:
            action = policy.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            timeout = memory.store(obs, action, reward, done)
            obs = next_obs
        
            if timeout and done:
                break
            if timeout: 
                memory.store_terminal_obs(obs)
                break
            if done:
                obs = env.reset()

        policy.update()

if __name__ == '__main__':
    main()





