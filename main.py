import gym
import numpy as np
import torch
from torch import nn
from arguments import get_args
import os
import wandb

from models import ValueNet, Policy
from storage import ReplayBuffer
from algs import PPO
from utils import NormalizedEnv
# TODO (add all tricks, normalization, gradient clipping, value loss clipping, entropy bonus, check all hyperparams 
# of kostrikov ppo)
# Code my own EnvNormalize wrapper so I actually understand it
# Where are my sources of randomness? ()

# Things I want to log: losses, gradients, reward \

# What was I doing? Was going to add random seed to start debugging. SHould be able to learn with only 300 samples per 
# iteration. Plot gradient for debugging.

# How is Kostrikov plotting reward?
# I think I really just need to 1) check my logic and signs, 2) implement the rest of the tricks


def main():
    wandb.login()
    wandb.init(project='ppo-setup2', monitor_gym=False)
    args = get_args()
    
    wandb.config.update(args)

    num_updates = -(-args.num_env_steps//args.num_steps) # ceil division to ensure total steps are met

    # env = gym.wrappers.Monitor(NormalizedEnv(gym.make(args.env_name)), 
    #                             'recordings', 
    #                             force=True,
    #                             video_callable=lambda i: False)

    
    env = NormalizedEnv(gym.make(args.env_name))

    torch.manual_seed(args.seed)
    env.seed(args.seed)
    replay_buffer = ReplayBuffer(args.num_steps, env.observation_space.high.shape, env.action_space.high.shape)
    policy = Policy(hidden_size=args.hidden_size,
                    hidden_layers=args.hidden_layers,
                    activation=args.activation,
                    action_dim=env.action_space.high.shape,
                    obs_dim=env.observation_space.high.shape,
                    deterministic=args.deterministic,
                    independent_std=args.independent_std)

    value_net = ValueNet(hidden_size=args.hidden_size,
                         hidden_layers=args.hidden_layers,
                         activation=args.activation,
                         obs_dim=env.observation_space.high.shape)

    agent = PPO(replay_buffer=replay_buffer,
                value_net=value_net,
                policy=policy,
                lr=args.lr,
                opt_alg=args.opt_alg,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                eps=args.ppo_eps,
                epochs=args.epochs,
                value_loss_coef=args.value_loss_coef)

    wandb.watch(policy, log='all')
    wandb.watch(value_net, log='all')

    for i in range(num_updates):
        obs = env.reset()

        while True:
            action = policy.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            wandb.log({'reward': info['real_reward']})
            timeout = replay_buffer.store(obs, action, reward, done, info)
            obs = next_obs
            if timeout and done:
                break
            if timeout:
                replay_buffer.bootstrap_reward(obs, value_net(obs).item())
                break
            if done:
                obs = env.reset()
        avg_rew = replay_buffer.real_rewards.mean()
        print('iter',i,'avg reward', avg_rew)
        wandb.log({'Average Sample Reward': avg_rew})
        agent.update(wandb)
        torch.save(policy.state_dict(), os.path.join(wandb.run.dir, 'policy.pt'))
        torch.save(value_net.state_dict(), os.path.join(wandb.run.dir, 'value_net.pt'))


if __name__ == '__main__':
    main()
