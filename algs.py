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
                epochs=20):
        self.replay_buffer = replay_buffer
        self.policy = policy
        self.value_net = value_net
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps = eps
        self.epochs = epochs

        opt_alg = opt_alg.lower()
        params = list(policy.parameters()) + list(value_net.parameters())
        if opt_alg == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=lr)
        elif opt_alg == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=lr)
        else:
            raise ValueError('\'%s\' is not currently a support optimization algorithm' %opt_alg)


    def update(self, wandb=None):
        ''' Processes the sample data and updates the policy and value networks. Returns data for logging.'''
        assert not self.policy.deterministic

        # do preprocessing
        with torch.no_grad():
            self.replay_buffer.compute_empirical_values(self.gamma)
            self.replay_buffer.compute_gae(self.value_net(self.replay_buffer.obs), self.gamma, self.gae_lambda)
            old_log_probs = self.policy.get_log_probs(self.replay_buffer.obs[:-1], self.replay_buffer.actions)

        # train the neural networks
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            ppo_clip_loss = self._compute_ppo_clip_loss(old_log_probs)
            value_loss = self._compute_value_loss()
            loss = -ppo_clip_loss + value_loss
            # print(loss)
            loss.backward()
            self.optimizer.step()

            # logging
            if wandb:
                wandb.log({'Value MSE Loss': value_loss,
                            'PPO Clip Loss': ppo_clip_loss})


    def _compute_ppo_clip_loss(self, old_log_probs):
        log_probs = self.policy.get_log_probs(self.replay_buffer.obs[:-1], self.replay_buffer.actions)
        ratios = (log_probs - old_log_probs).exp()
        return torch.min(ratios * self.replay_buffer.advantages, torch.clamp(ratios, 1 - self.eps, 1 + self.eps)\
            * self.replay_buffer.advantages).mean()


    def _compute_value_loss(self):
        '''Returns clipped version of MSE loss for value'''
        return (self.value_net(self.replay_buffer.obs[:-1]).squeeze() -\
            self.replay_buffer.empirical_values).pow(2).mean()

