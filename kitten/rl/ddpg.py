import torch
import torch.nn as nn
from torch import Tensor

from kitten.experience import AuxiliaryMemoryData, Transitions
from kitten.nn import (
    AddTargetNetwork,
    Actor,
    Critic,
    Value,
    HasActor,
    HasCritic,
    HasValue,
    CriticPolicyPair,
)
from kitten.rl import Algorithm
from kitten.common.typing import Log, ModuleNamePairs, Device


class DeepDeterministicPolicyGradient(
    Algorithm[AuxiliaryMemoryData], HasActor, HasCritic, HasValue
):
    """Implements DDPG

    Lillicrap et al. Continuous Control with Deep Reinforcement Learning. 2015.
    """

    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        gamma: float = 0.99,
        lr: float = 1e-3,
        tau: float = 0.005,
        clip_grad_norm: float | None = 1,
        update_frequency: int = 1,
        device: Device = "cpu",
        **kwargs,
    ):
        """DDPG

        Args:
            actor (nn.Module): Actor.
            critic (nn.Module): Critic.
            gamma (float, optional): Reward discount factor. Defaults to 0.99.
            lr (_type_, optional): Learning rate. Defaults to 1e-3.
            tau (float, optional): Target network update. Defaults to 0.005.
            policy_improvement_frequency (int): Steps between policy improvement.
            device (str, optional): Training hardware. Defaults to "cpu".
        """
        self._actor = AddTargetNetwork[Actor](actor_network, device=device)
        self._critic = AddTargetNetwork[Critic](critic_network, device=device)
        self._device = device

        self._gamma = gamma
        self._optim_actor = torch.optim.Adam(params=self._actor.net.parameters(), lr=lr)
        self._optim_critic = torch.optim.Adam(
            params=self._critic.net.parameters(), lr=lr
        )
        self._tau = tau
        self._clip_grad_norm = clip_grad_norm
        self._update_frequency = update_frequency

        self._loss_critic_value: float = 0.0
        self._loss_actor_value: float = 0.0

    @property
    def actor(self) -> Actor:
        return self._actor.net

    @property
    def critic(self) -> Critic:
        return self._critic.net

    @property
    def value(self) -> Value:
        return CriticPolicyPair(self.critic, self.policy_fn)

    def _critic_update(self, batch: Transitions, aux: AuxiliaryMemoryData) -> float:
        """Runs a critic update

        Args:
            s_0 (Tensor): Observation before action
            a (Tensor): Agent action
            r (Tensor): Reward (s_0, a) -> (s_1
            s_1 (Tensor): Observation after action
            d (Tensor): Dones
            weights (Tensor): Loss importance weighting for off-policy

        Returns:
            float: critic loss
        """
        td_error = (
            self.td_error(batch.s_0, batch.a, batch.r, batch.s_1, batch.d) * aux.weights
        )
        loss = torch.mean(td_error**2)
        loss_value = loss.item()
        self._optim_critic.zero_grad()
        loss.backward()
        if not self._clip_grad_norm is None:
            nn.utils.clip_grad_norm_(
                self._critic.net.parameters(), self._clip_grad_norm
            )
        self._optim_critic.step()

        return loss_value

    def td_error(self, s_0: Tensor, a: Tensor, r: Tensor, s_1: Tensor, d: Tensor):
        """Returns TD difference for a transition"""
        x = self._critic.net.q(s_0, a).squeeze()
        with torch.no_grad():
            target_max = (~d) * self._critic.target.q(
                s_1, self._actor.target.a(s_1)
            ).squeeze()
            y = r + target_max * self._gamma
        return y - x

    def _actor_update(self, s_0: Tensor, weights: Tensor) -> float:
        """Runs a critic update

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
        desired_action = self._actor.a(s_0)
        loss = -self._critic.q(s_0, desired_action)
        loss = torch.mean(
            torch.mean(loss, dim=list(range(1, len(loss.shape)))) * weights
        )
        loss_value = loss.item()
        self._optim_actor.zero_grad()
        loss.backward()
        if not self._clip_grad_norm is None:
            nn.utils.clip_grad_norm_(self._actor.net.parameters(), self._clip_grad_norm)
        self._optim_actor.step()

        return loss_value

    def update(
        self, batch: Transitions, aux: AuxiliaryMemoryData, step: int
    ) -> tuple[float, float]:
        """Runs a DDPG update step

        Args:
            batch (Tuple): Batch of training transitions from the replay buffer
            aux (AuxiliaryMemoryData): Recommended importance sampling weights among other things
            step (int): Current step of training

        Returns:
            Tuple[float, float]: Critic loss and actor loss
        """
        if step % self._update_frequency == 0:
            loss_critic_value = self._critic_update(batch, aux)
            loss_actor_value = self._actor_update(batch.s_0, aux.weights)
            self._critic.update_target_network(tau=self._tau)
            self._actor.update_target_network(tau=self._tau)

            self._loss_critic_value, self._loss_actor_value = (
                loss_critic_value,
                loss_actor_value,
            )
        return self._loss_critic_value, self._loss_actor_value

    def policy_fn(self, s: Tensor) -> Tensor:
        return self._actor.net.a(s)

    def get_log(self) -> Log:
        return {
            "critic_loss": self._loss_critic_value,
            "actor_loss": self._loss_actor_value,
        }

    def get_models(self) -> ModuleNamePairs:
        return [(self._actor.net, "actor"), (self._critic.net, "critic")]
