from typing import Callable

import torch
from torch import Tensor
import torch.nn as nn

from kitten.experience import AuxiliaryMemoryData, Transitions
from kitten.intrinsic.intrinsic import IntrinsicReward
from kitten.nn import Ensemble
from kitten.common.rng import Generator


class Disagreement(IntrinsicReward):
    """Self-Supervised Exploration via Disagreement

    Pathak et al. https://arxiv.org/pdf/1906.04161.pdf
    """

    def __init__(
        self,
        build_forward_head: Callable[[], nn.Module],
        feature_net: nn.Module | None = None,
        ensemble_number: int = 5,
        lr: float = 1e-3,
        discrete_action_space: bool = False,
        rng: Generator | None = None,
        **kwargs,
    ):
        """Creates the disagreement module"""
        super().__init__(**kwargs)
        self.feature_net = feature_net
        self.ensemble_number = ensemble_number
        self.forward_heads = Ensemble(
            build_forward_head,
            n=self.ensemble_number,
            rng=rng
        )
        self.discrete = discrete_action_space
        if self.discrete:
            raise NotImplementedError("Not yet implemented for discrete action space")
        self._optim = torch.optim.Adam(
            params=self.forward_heads.parameters(), lr=lr
        )

    def _featurise(self, s: Tensor) -> Tensor:
        """Encodes into a latent space

        Ideally, the latent space should be easy to learn from and relevant to the goal

        Args:
            s (Tensor): Observations

        Returns:
            Tensor: Encoded states
        """
        if not self.feature_net is None:
            return self.feature_net(s)
        return s

    def _update(self, batch: Transitions, aux: AuxiliaryMemoryData, step: int):
        self._optim.zero_grad()
        phi_0 = self._featurise(batch.s_0)
            # TODO: Rewrite with abstraction
            # Rather than cat
        pred_phi_1 = self.forward_heads(torch.cat((phi_0, batch.a), -1))
        true_phi_1 = self._featurise(batch.s_0).unsqueeze(0)

        # Train on different subsets of data (bootstrap)
        split = 1.0 / self.ensemble_number
        mask = torch.stack(
            [
                torch.where(
                    (i * split < aux.random) & (aux.random < (i + 1) * split), 1, 0
                )
                for i in range(self.ensemble_number)
            ]
        ).unsqueeze(-1)
        pred_phi_1 = pred_phi_1 * mask
        true_phi_1 = true_phi_1 * mask

        weights = aux.weights.unsqueeze(0).unsqueeze(-1)
        loss = torch.mean(((true_phi_1 - pred_phi_1) * weights) ** 2)
        self.info["loss"] = loss.item()
        loss.backward()
        self._optim.step()

    def _reward(self, batch: Transitions):
        phi_0 = self._featurise(batch.s_0)
        pred_phi_1 = self.forward_heads(torch.cat((phi_0, batch.a), -1))
        r_i = torch.mean(torch.var(pred_phi_1, dim=0), dim=-1)

        return r_i
