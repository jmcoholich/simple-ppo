import numpy as np
import torch


class PPO:
    def __init__(self, replay_buffer, policy, value_net, lr=3e-4, opt_alg='Adam'):
        self.replay_buffer = replay_buffer
        self.policy = policy
        self.value_net = value_net

        opt_alg = opt_alg.lower()
        params = list(policy.parameters()) + list(value_net.parameters())
        if opt_alg == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=lr)  
        elif opt_alg == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=lr)
        else:
            raise ValueError('\'%s\' is not currently a support optimization algorithm' %opt_alg)

    def update(self):
        # do preprocessing

        # start training the neural networks
        self.optmizer.zero_grad()
        loss = self._compute_ppo_clip_loss()
        loss.backward()
        self.optimizer.step()
        

    def _compute_ppo_clip_loss(self):
        pass
