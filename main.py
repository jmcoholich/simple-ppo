import gym
import numpy as np
import torch
from torch import nn
from arguments import get_args
import os
import wandb
import sys

from models import ValueNet, Policy
from storage import ReplayBuffer
from algs import PPO
from utils import NormalizedEnv
# TODO (add all tricks, gradient clipping, entropy bonus, check all hyperparams 
# of kostrikov ppo)
# Code my own EnvNormalize wrapper so I actually understand it
# Where are my sources of randomness? ()

# Things I want to log: losses, gradients, reward \

# What was I doing? Was going to add random seed to start debugging. SHould be able to learn with only 300 samples per 
# iteration. Plot gradient for debugging.

# How is Kostrikov plotting reward?
# I think I really just need to 1) check my logic and signs, 2) implement the rest of the tricks


# things to fix
# should I be normalizing or clipping my action outputs? 
# maybe do an ablation analysis where I take away the normalization.
# Look up RL sanity checks 
# Hack the EnvNormalize wrapper to make an extremely simple env to test in
# I know its not an optimziation problem, its somewhere else


# New steps to do 
# See if I can learn this simple env with the kostrikov implementation
# See what the normalized values of this env are 
# Ok but seriously what I need to do is comb through each file in order, line for line, checking for mistakes


# Combing through, things to double check
# - when I bootstrap the reward and store the final observation, make sure things are fine in either case. Make sure 
# nothing in the buffers carries over from previous iterations. I know the consequence of making the final reward zero 
# is fine. What about leaving the final observation zero? I would like to force the predicted value of the final observa
# tion to be zero if the timeout was done. Ultimately, I don't think I should worry about it because its rare to timeout
# and be done at the same time. Actually, I will let the value net calculate the value of the final obs bc sometimes 
# the env itself just times out. 

# Actually, not sure if I should bootstrap the final reward either...because my value function uses GAE lambda. 
# Ok...see what kostrikov does.
# He differentiates between true terminal states and timeout end states
# Kostrikov's reward buffer has shape N, not N + 1 like mine. Also check in compute_gae
# Perhaps start by ignoring the env timeout endstates -- just treat those like true terminal

# Ok now to-do
# - understand kostrikov's implementation of masks and bad masks, find out how he finds out if the env just timed out
# - implement that in my own code
# - try on cheese env, then pendulum


# Also to do-- store old log probs during the forward pass instead of having to calculate them again later
# 
# I* don't think I need to write a wrapper to keep track of truncate time limits, bc the gym envs already output when 
# they are truncated. Should I rely on the member variable or info? 

# I think kostrikov's implementation has the small hole that the env could reach a terminal state at the same time as 
# it times out.

# so I have env._max_episode_steps (seems like every env has this, and perhaps I should add timeout to my own?)
# and in info, once 'done' happens, I have {'TimeLimit.truncated': True}

# Now to-do ... just correctly bootstrap and calculate advantages and values, taking into account the truncation info
# I think calling env.reset() at the beginning of every sample is fine.

# Just add an assert statement to ensure that whenever I have done, I also have a truncation key in info





# Changes I will make
# - store self.env_timeout along with self.done and figure out how to shit myself.
# env timeout should also be true when my buffer gets full, because I will presumably be doing the same thing.
# - throw up a warning if I get a new type of info
def main():
    wandb.login()
    wandb.init(project='ppo-setup3', monitor_gym=False)
    args = get_args()
    wandb.config.update(args)
    num_updates = -(-args.total_steps//args.steps_per_update) # ceil division to ensure total steps are met

    # env = gym.wrappers.Monitor(NormalizedEnv(gym.make(args.env_name)), 
    #                             'recordings', 
    #                             force=True,
    #                             video_callable=lambda i: False)

    env = NormalizedEnv(gym.make(args.env_name), gamma=args.gamma)
    wandb.config.env_max_episode_steps = env._max_episode_steps
    # env = gym.make(args.env_name) #  
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    replay_buffer = ReplayBuffer(args.steps_per_update, env.observation_space.high.shape, env.action_space.high.shape)
    policy = Policy(action_dim=env.action_space.high.shape,
                    obs_dim=env.observation_space.high.shape,
                    hidden_size=args.hidden_size,
                    hidden_layers=args.hidden_layers,
                    activation=args.activation,
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
                value_loss_coef=args.value_loss_coef,
                entropy_coef=args.entropy_coef)

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
                replay_buffer.bootstrap_reward(obs, value_net(obs).item()) #TODO think about whether this is right or no
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