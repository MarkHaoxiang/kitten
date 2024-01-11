import torch
from torch import Tensor
import torch.nn as nn
from curiosity.experience import AuxiliaryMemoryData, Transition

from curiosity.intrinsic.intrinsic import IntrinsicReward
class IntrinsicCuriosityModule(IntrinsicReward, nn.Module):
    """ Intrinsic curiosity, intrinsic reward

    Pathak et al. https://arxiv.org/pdf/1705.05363.pdf
    """
    
    def __init__(self,
                 feature_net: nn.Module,
                 forward_head: nn.Module,
                 inverse_head: nn.Module,
                 beta: float = 0.2,
                 lr: float = 1e-3,
                 discrete_action_space: bool = False,
                 **kwargs):
        """ Creates the intrinsic curiosity module

        Args:
            feature_net (nn.Module): From observations to encodings
            forward_head (nn.Module): From encoding and action to encoding
            inverse_head (nn.Module): From two encodings to action
            beta (float, optional): Forward/Inverse loss scaling factor. Default 0.2.
        """
        super().__init__(**kwargs)
        self.feature_net = feature_net
        self.forward_head = forward_head
        self.inverse_head = inverse_head
        self.beta = beta
        self.discrete = discrete_action_space
        if self.discrete:
            raise NotImplementedError("Not yet implemented for discrete action space")
        self._optim = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward_model(self, s_0: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """ Predicts next encoding

        Args:
            s_0 (torch.Tensor): Current obs
            a (_type_): Current action

        Returns:
            torch.Tensor: Next encoding
        """
        batch_size = s_0.shape[:-1]
        phi_0 = self.feature_net(s_0)
        pred_phi_1 = self.forward_head(torch.cat((phi_0, a), len(batch_size)))
        return pred_phi_1
    
    def inverse_model(self, s_0: torch.Tensor, s_1: torch.Tensor) -> torch.Tensor:
        """Predicts action

        Args:
            s_0 (torch.Tensor): Current obs
            s_1 (torch.Tensor): Next obs

        Returns:
            torch.Tensor: Action
        """
        batch_size = s_0.shape[:-1]
        phi_0 = self.feature_net(s_0)
        phi_1 = self.feature_net(s_1)
        a = self.inverse_head(torch.cat((phi_0, phi_1), len(batch_size)))
        return a

    def forward(self, s_0: torch.Tensor, s_1: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """ Calculates the intrinsic reward

        Args:
            s_0 (torch.Tensor): Current obs
            s_1 (torch.Tensor): Next obs
            a (torch.Tensor): Action

        Returns:
            torch.Tensor: Intrinsic reward
        """
        pred_phi_1 = self.forward_model(s_0, a)
        true_phi_1 = self.feature_net(s_1)
        r_i = torch.linalg.vector_norm(true_phi_1-pred_phi_1, dim=(-1)) / 2
        return r_i
    
    def _calc_loss(self, s_0: torch.Tensor, s_1: torch.Tensor, a: torch.Tensor, weights: torch.Tensor):
        pred_phi_1 = self.forward_model(s_0, a)
        true_phi_1 = self.feature_net(s_1)
        weights = weights.unsqueeze(-1)
        forward_loss = 0.5 * torch.mean(((true_phi_1-pred_phi_1) * weights) ** 2)
        self.info["forward_loss"] = forward_loss.item()
        pred_a = self.inverse_model(s_0, s_1)
        if self.discrete:
            raise NotImplementedError("Not yet implemented for discrete action space")
        else:
            inverse_loss = torch.mean(((a-pred_a) * weights) ** 2)
            self.info["inverse_loss"] = inverse_loss.item()
        return forward_loss * self.beta + inverse_loss * (1-self.beta) 

    def _update(self, s_0: torch.Tensor, s_1: torch.Tensor, a: torch.Tensor, weights: torch.Tensor):
        self._optim.zero_grad()
        loss = self._calc_loss(s_0, s_1, a, weights)
        loss.backward()
        self._optim.step()

    def update(self, batch: Transition, aux: AuxiliaryMemoryData, step: int):
        return self._update(batch.s_0, batch.s_1, batch.a, aux.weights)

    def _reward(self, batch: Transition):
        return self.forward(batch.s_0, batch.s_1, batch.a)
