from typing import Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn
from curiosity.experience import Transition
from curiosity.intrinsic.intrinsic import IntrinsicReward

class IntrinsicCuriosityModule(IntrinsicReward):
    """ Self-Supervised Exploration via Disagreement

    Pathak et al. https://arxiv.org/pdf/1906.04161.pdf
    """
    
    def __init__(self,
                 build_forward_head: Callable,
                 feature_net: Optional[nn.Module] = None,
                 ensemble_number: int = 5,
                 lr: float = 1e-3,
                 discrete_action_space: bool = False,
                 **kwargs):
        """ Creates the disagreement module
        """
        super().__init__()
        self.feature_net = feature_net
        self.forward_heads = [build_forward_head() for _ in range(ensemble_number)]
        self.discrete = discrete_action_space
        if self.discrete:
            raise NotImplementedError("Not yet implemented for discrete action space")
        self._optim = torch.optim.Adam(
            params=nn.ModuleList(self.forward_heads).parameters(),
            lr=lr
        )

    def _forward_models(self, phi_0: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """ Predicts next encodings
        """
        batch_size = phi_0.shape[:-1]
        return torch.stack([h(torch.cat((phi_0, a), len(batch_size)))for h in self.forward_heads])

    def _featurise(self, s):
        if self.feature_net is None:
            return self.feature_net(s)
        return s 

    def update(self, batch: Transition, weights: Tensor, step: int):
        self._optim.zero_grad()
        phi_0 = self._featurise(batch.s_0)
        pred_phi_1 = self._forward_models(phi_0, batch.a)
        true_phi_1 = self._featurise(batch.s_0).unsqueeze(0)
        weights = weights.unsqueeze(0).unsqueeze(-1)
        loss = torch.mean(((true_phi_1-pred_phi_1) * weights) ** 2)
        self.info['loss'] = loss.item()
        loss.backward()
        self._optim.step()

    def _reward(self, batch: Transition):
        phi_0 = self._featurise(batch.s_0)
        pred_phi_1 = self._forward_models(phi_0, batch.a)
        r_i = torch.mean(torch.var(pred_phi_1, dim=0))
        return r_i