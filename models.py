import numpy as np
import torch
from torch import nn

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


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_size=256, hidden_layers=2, activation='relu'):
        super(ValueNet, self).__init__()

        obs_dim, = obs_dim
        activation = activation.lower()

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
        self.linear_layers.append(nn.Linear(hidden_size, 1))

    def forward(self, obs):
        obs = torch.from_numpy(obs).float()
        for i in range(len(self.linear_layers) - 1):
            obs = self.act(self.linear_layers[i](obs))
        return self.linear_layers[-1](obs)
