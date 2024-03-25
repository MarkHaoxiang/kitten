import torch
from torch import Tensor
import torch.nn as nn
from kitten.experience import AuxiliaryMemoryData, Transitions

from kitten.intrinsic.intrinsic import IntrinsicReward
from kitten.experience import Transitions


class RandomNetworkDistillation(IntrinsicReward, nn.Module):
    """Exploration by Random Network Distillation

    Burda et al. https://arxiv.org/pdf/1810.12894.pdf
    """

    def __init__(
        self,
        target_net: nn.Module,
        predictor_net: nn.Module,
        lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_net = target_net
        self.predictor_net = predictor_net
        self._optim = torch.optim.Adam(params=self.predictor_net.parameters(), lr=lr)

        # target_net does not require training
        for param in self.target_net.parameters():
            param.requires_grad = False

    def forward(self, s_1: Tensor) -> Tensor:
        """Calculates the intrinsic reward

        Args:
            s_1 (Tensor): Next obs

        Returns:
            Tensor: Intrinsic reward
        """
        random_embedding = self.target_net(s_1)
        predicted_embedding = self.predictor_net(s_1)
        r_i = ((random_embedding - predicted_embedding) ** 2).mean(-1)
        return r_i

    def _update(self, batch: Transitions, aux: AuxiliaryMemoryData, step: int):
        # Train Predictor Network
        random_embedding = self.target_net(batch.s_1)
        predicted_embedding = self.predictor_net(batch.s_1)
        weights = aux.weights.unsqueeze(-1)

        self._optim.zero_grad()
        loss = torch.mean(((random_embedding - predicted_embedding) * weights) ** 2)
        self.info["loss"] = loss.item()
        loss.backward()
        self._optim.step()

    def _reward(self, batch: Transitions):
        return self.forward(batch.s_1)
