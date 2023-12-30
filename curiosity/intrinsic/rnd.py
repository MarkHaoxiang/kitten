import torch
from torch import Tensor
import torch.nn as nn

from curiosity.intrinsic.intrinsic import IntrinsicReward
from curiosity.experience import Transition

class RandomNetworkDistillation(nn.Module, IntrinsicReward):
    """ Exploration by Random Network Distillation

    Burda et al. https://arxiv.org/pdf/1810.12894.pdf
    """
    def __init__(self,
                 target_net: nn.Module,
                 predictor_net: nn.Module,
                 lr: float = 1e-3,
                 **kwargs):
        super().__init__()
        self.target_net = target_net
        self.predictor_net = predictor_net
        self._optim = torch.optim.Adam(params=self.predictor_net.parameters(), lr=lr)

    def forward(self, s_1: Tensor) -> Tensor:
        """ Calculates the intrinsic reward

        Args:
            s_1 (Tensor): Next obs

        Returns:
            Tensor: Intrinsic reward
        """
        random_embedding = self.target_net(s_1)
        predicted_embedding = self.predictor_net(s_1)
        r_i = ((random_embedding - predicted_embedding)**2).sum(-1)
        return r_i
    
    def _update(self, s_1: Tensor, weights: Tensor):
        random_embedding = self.target_net(s_1)
        predicted_embedding = self.predictor_net(s_1)
        weights = weights.unsqueeze(-1)
    
        self._optim.zero_grad()
        loss = torch.mean(((random_embedding - predicted_embedding) * weights)**2)
        self.info['loss'] = loss.item()
        loss.backward()
        self._optim.step()

    def update(self, batch: Transition, weights: Tensor, step: int):
        return self._update(batch.s_1, weights)

    def _reward(self, batch: Transition):
        return self.forward(batch.s_1)
