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
# TODO ( add loss function and training, add normalization)


def main():
    wandb.login()
    wandb.init(project='ppo-setup')
    args = get_args()
    wandb.config.update(args)

    num_updates = -(-args.num_env_steps//args.num_steps) # ceil division to ensure total steps are met

    env = gym.make(args.env_name)
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
                epochs=args.epochs)

    wandb.watch(policy)
    wandb.watch(value_net)


    for i in range(num_updates):
        obs = env.reset()

        while True:
            action = policy.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            timeout = replay_buffer.store(obs, action, reward, done)
            obs = next_obs

            if timeout and done:
                break
            if timeout:
                replay_buffer.bootstrap_reward(obs, value_net(obs).item())
                break
            if done:
                obs = env.reset()
        print('iter',i,'avg reward', replay_buffer.rewards.mean())
        logs = agent.update()
        wandb.log(logs)
        # breakpoint()
        print('LOGGED' * 20)

if __name__ == '__main__':
    main()
