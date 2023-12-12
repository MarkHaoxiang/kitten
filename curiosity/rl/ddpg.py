from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from curiosity.nn import AddTargetNetwork

class DeepDeterministicPolicyGradient:
    """ Implements DDPG loss
    """
    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 gamma: float = 0.99,
                 lr: float=1e-3,
                 tau: float = 0.005,
                 clip_grad_norm: Optional[float] = 1,
                 device: str = "cpu",
                 **kwargs):
        """ DDPG

        Args:
            actor (nn.Module): Actor.
            critic (nn.Module): Critic.
            gamma (float, optional): Reward discount factor. Defaults to 0.99.
            lr (_type_, optional): Learning rate. Defaults to 1e-3.
            tau (float, optional): Target network update. Defaults to 0.005.
            device (str, optional): Training hardware. Defaults to "cpu".
        """
        self.actor = AddTargetNetwork(actor, device=device)
        self.critic = AddTargetNetwork(critic, device=device)
        self.device = device

        self._gamma = gamma
        self._optim_actor = torch.optim.Adam(params=self.actor.net.parameters(), lr=lr)
        self._optim_critic = torch.optim.Adam(params=self.critic.net.parameters(), lr=lr)
        self._tau = tau
        self._clip_grad_norm = clip_grad_norm

    def _critic_update(self, s_0: Tensor, a: Tensor, r: Tensor, s_1: Tensor, d: Tensor, weights: Tensor) -> float:
        """ Runs a critic update

        Args:
            s_0 (Tensor): Observation before action
            a (Tensor): Agent action
            r (Tensor): Reward (s_0, a) -> (s_1)
            s_1 (Tensor): Observation after action
            d (Tensor): Dones
            weights (Tensor): Loss importance weighting for off-policy

        Returns:
            float: critic loss
        """
        td_error = self.td_error(s_0, a, r, s_1, d) * weights
        loss = torch.mean(td_error**2)
        loss_value = loss.item()
        self._optim_critic.zero_grad()
        loss.backward()
        if not self._clip_grad_norm is None:
            nn.utils.clip_grad_norm_(self.critic.net.parameters(), self._clip_grad_norm)
        self._optim_critic.step()

        return loss_value
    
    def td_error(self, s_0: Tensor, a: Tensor, r: Tensor, s_1: Tensor, d: Tensor) -> float:
        """ Returns TD difference for a transition
        """
        x = self.critic(torch.cat((s_0, a), 1)).squeeze()
        with torch.no_grad():
            target_max = (~d) * self.critic.target(torch.cat((s_1, self.actor.target(s_1)), 1)).squeeze()
            y = (r + target_max * self._gamma)
        return y-x

    def _actor_update(self, s_0: Tensor, weights: Tensor) -> float:
        """ Runs a critic update

        Args:
            s_0 (Tensor): Observation before action
            a (Tensor): Agent action
            r (Tensor): Reward (s_0, a) -> (s_1)
            s_1 (Tensor): Observation after action
            d (Tensor): Dones
            weights (Tensor): Loss importance weighting for off-policy

        Returns:
            float: actor loss
        """
        desired_action = self.actor(s_0)
        loss = -self.critic(torch.cat((s_0, desired_action), 1))
        loss = torch.mean(torch.mean(loss, dim=list(range(1, len(loss.shape)))) * weights)
        loss_value = loss.item()
        self._optim_actor.zero_grad()
        loss.backward()
        if not self._clip_grad_norm is None:
            nn.utils.clip_grad_norm_(self.actor.net.parameters(), self._clip_grad_norm)
        self._optim_actor.step()

        return loss_value
    
    def update(self, batch: Tuple) -> Tuple[float, float]:
        """ Runs a DDPG update step

        Args:
            batch (Tuple): Batch of training transitions from the replay buffer

        Returns:
            Tuple[float, float]: Critic loss and actor loss
        """
        transitions, weights = batch
        loss_critic_value = self._critic_update(*transitions, weights)
        loss_actor_value = self._actor_update(transitions[0], weights)
        self.critic.update_target_network(tau=self._tau)
        self.actor.update_target_network(tau=self._tau)

        return loss_critic_value, loss_actor_value
