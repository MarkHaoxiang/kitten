from typing import List, Optional, Tuple, Union
from numpy import ndarray

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module

from curiosity.experience import AuxiliaryMemoryData, Transition
from curiosity.nn import AddTargetNetwork, Actor, Critic
from curiosity.rl import Algorithm, HasCritic

class DeepDeterministicPolicyGradient(Algorithm, HasCritic):
    """ Implements DDPG

    Lillicrap et al. Continuous Control with Deep Reinforcement Learning. 2015.
    """

    def __init__(self,
                 actor_network: Actor,
                 critic_network: Critic,
                 gamma: float = 0.99,
                 lr: float = 1e-3,
                 tau: float = 0.005,
                 clip_grad_norm: Optional[float] = 1,
                 update_frequency: int = 1,
                 device: str = "cpu",
                 **kwargs):
        """ DDPG

        Args:
            actor (nn.Module): Actor.
            critic (nn.Module): Critic.
            gamma (float, optional): Reward discount factor. Defaults to 0.99.
            lr (_type_, optional): Learning rate. Defaults to 1e-3.
            tau (float, optional): Target network update. Defaults to 0.005.
            policy_improvement_frequency (int): Steps between policy improvement.
            device (str, optional): Training hardware. Defaults to "cpu".
        """
        self.actor = AddTargetNetwork(actor_network, device=device)
        self._critic = AddTargetNetwork(critic_network, device=device)
        self.device = device

        self._gamma = gamma
        self._optim_actor = torch.optim.Adam(params=self.actor.net.parameters(), lr=lr)
        self._optim_critic = torch.optim.Adam(params=self.critic.net.parameters(), lr=lr)
        self._tau = tau
        self._clip_grad_norm = clip_grad_norm
        self._update_frequency = update_frequency

        self.loss_critic_value, self.loss_actor_value = 0, 0

    @property
    def critic(self):
        return self._critic
    
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

    def td_error(self, s_0: Tensor, a: Tensor, r: Tensor, s_1: Tensor, d: Tensor):
        """ Returns TD difference for a transition
        """
        x = self.critic.q(s_0, a).squeeze()
        with torch.no_grad():
            target_max = (~d) * self.critic.target.q(s_1, self.actor.target.a(s_1)).squeeze()
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
        desired_action = self.actor.a(s_0)
        loss = -self.critic.q(s_0, desired_action)
        loss = torch.mean(torch.mean(loss, dim=list(range(1, len(loss.shape)))) * weights)
        loss_value = loss.item()
        self._optim_actor.zero_grad()
        loss.backward()
        if not self._clip_grad_norm is None:
            nn.utils.clip_grad_norm_(self.actor.net.parameters(), self._clip_grad_norm)
        self._optim_actor.step()

        return loss_value

    def update(self, batch: Transition, aux: AuxiliaryMemoryData, step: int) -> Tuple[float, float]:
        """ Runs a DDPG update step

        Args:
            batch (Tuple): Batch of training transitions from the replay buffer
            aux (AuxiliaryMemoryData): Recommended importance sampling weights among other things
            step (int): Current step of training

        Returns:
            Tuple[float, float]: Critic loss and actor loss
        """
        if step % self._update_frequency == 0:
            loss_critic_value = self._critic_update(*batch, aux.weights)
            loss_actor_value = self._actor_update(batch.s_0, aux.weights)
            self.critic.update_target_network(tau=self._tau)
            self.actor.update_target_network(tau=self._tau)

            self.loss_critic_value, self.loss_actor_value = loss_critic_value, loss_actor_value
        return self.loss_critic_value, self.loss_actor_value 

    def policy_fn(self, s: Union[Tensor, ndarray]) -> Tensor:
        if isinstance(s, ndarray):
            s = torch.tensor(s, device=self.device)
        return self.actor.a(s)

    def get_log(self):
        return {
            "critic_loss": self.loss_critic_value,
            "actor_loss": self.loss_actor_value
        }

    def get_models(self) -> List[Tuple[Module, str]]:
        return [
            (self.actor.net, "actor"),
            (self.critic.net, "critic")
        ]
