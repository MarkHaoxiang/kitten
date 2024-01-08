import warnings

import torch
from torch import Tensor
import torch.nn as nn
from curiosity.experience import Transition

from curiosity.intrinsic.intrinsic import IntrinsicReward
from curiosity.experience import Transition
from curiosity.util.stat import RunningMeanVariance

class RandomNetworkDistillation(IntrinsicReward, nn.Module):
    """ Exploration by Random Network Distillation

    Burda et al. https://arxiv.org/pdf/1810.12894.pdf
    """
    def __init__(self,
                 target_net: nn.Module,
                 predictor_net: nn.Module,
                 lr: float = 1e-3,
                 normalised_obs_clip: float = 5,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_net = target_net
        self.predictor_net = predictor_net
        self._optim = torch.optim.Adam(params=self.predictor_net.parameters(), lr=lr)

        self.obs_normalisation = RunningMeanVariance()
        self.reward_normalisation = RunningMeanVariance()
        self._normalised_obs_clip = normalised_obs_clip

        # target_net does not require training
        for param in self.target_net.parameters():
            param.requires_grad = False

    def forward(self, s_1: Tensor) -> Tensor:
        """ Calculates the intrinsic reward

        Args:
            s_1 (Tensor): Next obs

        Returns:
            Tensor: Intrinsic reward
        """
        s_1 = (s_1 - self.obs_normalisation.mean.unsqueeze(0)) / self.obs_normalisation.std.unsqueeze(0)
        s_1 = torch.clamp(s_1, -self._normalised_obs_clip, self._normalised_obs_clip)

        random_embedding = self.target_net(s_1)
        predicted_embedding = self.predictor_net(s_1)
        r_i = ((random_embedding - predicted_embedding)**2).sum(-1) / 2
        return r_i

    def update(self, batch: Transition, weights: Tensor, step: int):
        # Update Obs Normalisation
        if torch.max(weights) != torch.min(weights):
            warnings.warn("Weighted sample normalisation is not yet implemented. Assuming weights are equal.")           
        self.obs_normalisation.add_tensor_batch(batch.s_1)

        # Update Reward Normalisation
        rewards = self.forward(batch.s_1)
        self.reward_normalisation.add_tensor_batch(rewards)

        # Train Predictor Network
        random_embedding = self.target_net(batch.s_1)
        predicted_embedding = self.predictor_net(batch.s_1)
        weights = weights.unsqueeze(-1)
    
        self._optim.zero_grad()
        loss = torch.mean(((random_embedding - predicted_embedding) * weights)**2)
        self.info['loss'] = loss.item()
        loss.backward()
        self._optim.step()

    def initialise(self, batch: Transition):
        super().initialise(batch)
        self.obs_normalisation.add_tensor_batch(batch.s_1)
        rewards = self.forward(batch.s_1)
        self.reward_normalisation.add_tensor_batch(rewards)

    def _reward(self, batch: Transition):
        return self.forward(batch.s_1) / self.reward_normalisation.std.unsqueeze(0)
