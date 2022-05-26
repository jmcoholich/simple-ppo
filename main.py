import gym
import numpy as np
import torch
from torch import nn
from arguments import get_args
import os
from datetime import datetime
import wandb
import sys
import pickle

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

# Check to make sure my real_rewards logging isn't messed up by adding the zero

# ALSO make sure I'm back propping through the correct things!! Make sure I call torch.no_grad() on the correct things.

# So... I need to ensure that my value loss and entropy loss aren't using the bootstrapped steps
# But let me think. The existence of the bootstrapped step causes lots of problems. But I _do_ need it in order to store
# final state to get the value of it... but do I actually need to do that? how does kostrikov do it?


# Kostrikov does nothing with masks when calculating his loss functions, perhaps because he he already dealt with it
# when generating batches. Or, different scheme entirely, check gae lambda calc and batch generation

# !!! if all else fails, I found a major error in previous implementation!! my GAE had was adding delta[i+1] instead
# of self.advantages[i+1]

# So I don't think Kostrikov does what I'm doing, but I think i can make what I'm doing work, then try to clean it up
# later and think of a simpler way.


# Changes I will make
# - throw up a warning if I get a new type of info


""" Now what to do:
Run this implementation, see if it works, on original Pendulum.
Run implementation before all these changes, but fix the gae calculation error, see if it works on original
pendulum.
If only the second one works, think of a more simple way to take into account time_limits.  """
"""I think the way to make it work without the bootstrapped step is to add V(final obs) * gamma to the reward for that
step. Do the math to make sure that works."""


def main():
    args = get_args()
    play = args.play != "None"
    if args.wandb:
        wandb.login()
        wandb.init(project="ppo-setup3", monitor_gym=False)
        wandb.config.update(args)
    else:
        wandb = None
    if not os.path.isdir("saved_models"):
        os.mkdir("saved_models")
    if play:
        datetime_str = args.play
    else:
        datetime_str = datetime.now().strftime("%y%m%d%H%M%S%f")
    save_folder = os.path.join("saved_models", datetime_str)
    if not play:
        os.mkdir(save_folder)
        print(f"Saving model checkpoints in {save_folder}")
    else:
        print(f"Loading model checkpoint from {save_folder}")

    max_rew = -1000000.0
    num_updates = -(
        -args.total_steps // args.steps_per_update
    )  # ceil division to ensure total steps are met

    # env = gym.wrappers.Monitor(NormalizedEnv(gym.make(args.env_name)),
    #                             'recordings',
    #                             force=True,
    #                             video_callable=lambda i: False)
    if play:
        with open(os.path.join(save_folder, "env_norm.pkl"), "rb") as f:
            env = pickle.load(f)
    else:
        env = gym.make(args.env_name)
        env = NormalizedEnv(env, gamma=args.gamma)

    if args.wandb:
        wandb.config.env_max_episode_steps = env.env._max_episode_steps
    # env = gym.make(args.env_name) #
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    replay_buffer = ReplayBuffer(
        args.steps_per_update,
        env.observation_space.high.shape,
        env.action_space.high.shape,
    )
    policy = Policy(
        action_dim=env.action_space.high.shape,
        obs_dim=env.observation_space.high.shape,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        deterministic=args.deterministic,
        independent_std=args.independent_std,
    )

    value_net = ValueNet(
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        obs_dim=env.observation_space.high.shape,
    )

    agent = PPO(
        replay_buffer=replay_buffer,
        value_net=value_net,
        policy=policy,
        lr=args.lr,
        opt_alg=args.opt_alg,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        eps=args.ppo_eps,
        epochs=args.epochs,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
    )
    if args.play != "None":
        # LOAD the policy state dict
        if args.play[-3:] != ".pt":
            args.play += ".pt"
        path = os.path.join(save_folder, "policy.pt")
        saved_state_dict = torch.load(path)
        policy.load_state_dict(saved_state_dict, strict=True)
    if args.wandb:
        wandb.watch(policy, log="all")
        wandb.watch(value_net, log="all")
    for i in range(num_updates):
        obs = env.reset()
        while True:
            if args.play != "None":
                env.render()
            action = policy.get_action(obs, deterministic=play)
            next_obs, reward, done, info = env.step(action)
            if args.wandb:
                wandb.log({"reward": info["real_reward"]})
            buffer_timeout = replay_buffer.store(obs, action, reward, done, info)
            obs = next_obs

            # an episode always ends with done = True, no matter if it occured through the env returning done (which
            # occurs for terminal state and for env timeout) or for the buffer filling up and timing out
            # if timeout=True as well, then the step after done=True will contain a bootstrapped reward. Else, 0 reward.
            if done:
                if info["TimeLimit.truncated"]:
                    with torch.no_grad():
                        reward = value_net(
                            obs
                        ).item()  # value net should already be outputting normalized values
                        real_reward = env.inverse_filt_rew(reward).item()
                else:
                    reward = 0  # rew = 0 normalized by env wrapper is still 0, since mean is not subtracted
                    real_reward = 0
                # in this stored step, the 'action' and 'done' values should never be used, they are just placeholders
                replay_buffer.store(
                    obs, action, reward, False, {"real_reward": real_reward}
                )
                obs = env.reset()
            else:
                assert not "TimeLimit.truncated" in info
            if buffer_timeout:
                if (
                    done
                ):  # I have already taken care of the stuff that needs to happen, (N+1)th values filled
                    break
                with torch.no_grad():
                    reward = value_net(obs).item()
                    real_reward = env.inverse_filt_rew(reward).item()
                replay_buffer.store(
                    obs, action, reward, False, {"real_reward": real_reward}
                )
                break

            # if timeout and done:
            #     replay_buffer.bootstrap_reward(obs, value_net(obs).item()) #TODO think about whether this is right or no
            #     break
            # if timeout:
            #     replay_buffer.bootstrap_reward(obs, value_net(obs).item())
            #     break
            # if done:
            #     obs = env.reset()
        avg_rew = replay_buffer.real_rewards.mean()
        print(f"iter: {i} | Average Reward: {avg_rew.item():0.2f}")
        if args.play == "None":
            if args.wandb:
                wandb.log({"Average Sample Reward": avg_rew})
            agent.update(wandb)
            if avg_rew > max_rew:
                path = os.path.join(save_folder, "policy.pt")
                print(f"==> Saving policy with reward {avg_rew :0.2f} as {path}")
                torch.save(policy.state_dict(), path)
                with open(os.path.join(save_folder, "env_norm.pkl"), "wb") as f:
                    pickle.dump(env, f)
                max_rew = avg_rew
            # torch.save(value_net.state_dict(), os.path.join(wandb.run.dir, "value_net.pt"))


if __name__ == "__main__":
    main()
