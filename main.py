import gym 
import numpy as np
import torch
from torch import nn
from arguments import get_args


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
            return True # return value indicates timeout
        else:
            return False


    def store_terminal_obs(self, obs):
        assert self.idx == self.N
        self.obs[self.idx] = obs


class Policy(nn.Module):
    def __init__(self, hidden_size=256, hidden_layers=2, activation='relu'):
        super(Policy, self).__init__() 
        
        

    

    def get_action(self, obs):
        return np.array([4 * (np.random.rand() - 0.5)])



class PPO:
    def __init__(self):
        pass


def main():
    args = get_args()

    num_updates = -(-args.num_env_steps//args.num_steps) # ceil division to ensure total steps are met

    env = gym.make('Pendulum-v0')
    memory = ReplayBuffer(args.num_steps, env.observation_space.high.shape, env.action_space.high.shape)
    policy = Policy(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers, activation=args.activation)


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


        


