import torch


class PPO:
    def __init__(self, 
                replay_buffer, 
                policy, 
                value_net, 
                lr=3e-4,
                opt_alg='Adam', 
                gamma=0.99, 
                gae_lambda=0.95, 
                eps=0.2,
                epochs=10,
                value_loss_coef=0.5,
                entropy_coef=0.0):
        self.replay_buffer = replay_buffer
        self.policy = policy
        self.value_net = value_net
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps = eps
        self.epochs = epochs
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        opt_alg = opt_alg.lower()
        params = list(policy.parameters()) + list(value_net.parameters())
        if opt_alg == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=lr)
        elif opt_alg == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=lr)
        else:
            raise ValueError('\'%s\' is not currently a support optimization algorithm' %opt_alg)

        self.epoch_counter = 0 # used for wandb logging


    def update(self, wandb=None):
        ''' Processes the sample data and updates the policy and value networks. Returns data for logging.'''
        assert not self.policy.deterministic

        # do preprocessing
        with torch.no_grad():
            old_predicted_values = self.value_net(self.replay_buffer.obs).squeeze()
            self.replay_buffer.compute_gae(old_predicted_values, self.gamma, self.gae_lambda) # checked 
            # below is fine, just don't use supp
            value_targets = self.replay_buffer.advantages + old_predicted_values[:-1] 
            old_log_probs, _  = self.policy.get_log_probs(self.replay_buffer.obs[:-1], self.replay_buffer.actions[:-1], 
                                                            compute_entropy=False) # again just don't use supp
            # calculation of mean exludes the 'bootstrapped' steps at end of every episode 
            N = len(self.replay_buffer.advantages)
            bs_idcs = self.replay_buffer.done.nonzero() + 1
            if N in bs_idcs:
                assert bs_idcs[-1,0] == N
                bs_idcs = bs_idcs[:-1,:] # get rid of that last value, since it's already gone
            mean_ = (self.replay_buffer.advantages.sum() - self.replay_buffer.advantages[bs_idcs].sum())/\
                    (N - len(bs_idcs))
            advantages = self.replay_buffer.advantages - mean_
            advantages /= self.replay_buffer.advantages.std() + 1e-9
            # ensure that the advantage at any bootstrapped step is zero, so that nothing backprops through it
            advantages[bs_idcs] = 0.0

        # train the neural networks
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            ppo_clip_obj, entropy = self._compute_ppo_clip_obj(old_log_probs, advantages, bs_idcs,
                compute_entropy=True) #(0.0 != self.entropy_coef)) # Always compute entropy just for logging 
            value_loss = self._compute_value_loss(value_targets, old_predicted_values[:-1], bs_idcs)
            loss = -ppo_clip_obj + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            loss.backward()
            self.optimizer.step()

            # logging
            if wandb:
                wandb.log({'Value MSE Loss': value_loss,
                            'PPO Clip Objective': ppo_clip_obj,
                            'epoch': self.epoch_counter,
                            'Average Entropy': entropy})
                self.epoch_counter += 1


    def _compute_ppo_clip_obj(self, old_log_probs, advantages, bs_idcs, compute_entropy=True):
        '''Combine with entropy calculation to avoid separate fwd pass for entropy'''
        log_probs, entropy = self.policy.get_log_probs(self.replay_buffer.obs[:-1], self.replay_buffer.actions[:-1],
            compute_entropy=compute_entropy)
        ratios = (log_probs - old_log_probs).exp()
        # calculate mean of entropy without including bs steps
        assert len(entropy) == self.replay_buffer.N
        entropy_mean = (entropy.sum() - entropy[bs_idcs].sum())/(len(entropy) - len(bs_idcs))
        return torch.min(ratios * advantages, torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages).mean(),\
            entropy_mean


    def _compute_value_loss(self, value_targets, old_predicted_values, bs_idcs):
        '''Returns clipped version of MSE loss for value'''
        current_value_predictions = self.value_net(self.replay_buffer.obs[:-1]).squeeze()
        value_loss = (current_value_predictions - value_targets).pow(2)
        value_pred_clipped = old_predicted_values + torch.clamp(current_value_predictions - old_predicted_values,
            -self.eps, self.eps)
        clipped_value_loss  = (value_pred_clipped - value_targets).pow(2)
        loss_ = torch.max(value_loss, clipped_value_loss)
        assert len(loss_) != 1
        return 0.5 * (loss_.sum() - loss_[bs_idcs].sum())/(len(loss_) - len(bs_idcs))