import torch
from torch import Tensor
import torch.nn as nn


class RandomNetworkDistillation(nn.Module):
    """ Exploration by Random Network Distillation

    Burda et al. https://arxiv.org/pdf/1810.12894.pdf
    """
    def __init__(self,
                 target_net: nn.Module,
                 predictor_net: nn.Module,
                 lr: float = 1e-3):
        super().__init__()
        self.target_net = target_net
        self.predictor_net = predictor_net
        self.info = {}

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
    
    def update(self, s_1: Tensor, weights: Tensor):
        random_embedding = self.target_net(s_1)
        predicted_embedding = self.predictor_net(s_1)
    
        self._optim.zero_grad()
        loss = torch.mean(((random_embedding - predicted_embedding) * weights)**2)
        self.info['loss'] = loss.item()
        loss.backward()
        self._optim.step()

    def get_log(self):
        return self.info
