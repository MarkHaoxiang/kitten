import torch
import torch.nn as nn
from torch import Tensor
from kitten.experience import AuxiliaryMemoryData, Transitions

from kitten.nn import Actor, Critic, AddTargetNetwork
from kitten.rl.ddpg import DeepDeterministicPolicyGradient

from kitten.common.typing import Device


class TwinDelayedDeepDeterministicPolicyGradient(DeepDeterministicPolicyGradient):
    """TD3

    Twin Delayed Deep Deterministic Policy Gradient (DDPG with improvements)
    """

    def __init__(
        self,
        actor_network: Actor,
        critic_1_network: Critic,
        critic_2_network: Critic,
        env_action_scale: Tensor,
        env_action_min: Tensor,
        env_action_max: Tensor,
        gamma: float = 0.99,
        lr: float = 1e-3,
        tau: float = 0.005,
        clip_grad_norm: float | None = 1,
        target_noise: float = 0.1,
        target_noise_clip: float = 0.2,
        critic_update_frequency: int = 1,
        policy_update_frequency: int = 2,
        device: Device = "cpu",
        **kwargs,
    ):
        """TD3

        Args:
            actor (nn.Module): Actor.
            critic_1 (nn.Module): Critic.
            critic_2 (nn.Module): Critic.
            gamma (float, optional): Reward discount factor. Defaults to 0.99.
            lr (_type_, optional): Learning rate. Defaults to 1e-3.
            tau (float, optional): Target network update. Defaults to 0.005.
            target_noise (float): Target network smoothing noise variance. Defaults to 0.1.
            target_noise_clip (float): Target network smoothing noise clip. Defaults to 0.2.
            device (str, optional): Training hardware. Defaults to "cpu".
        """
        super().__init__(
            actor_network=actor_network,
            critic_network=critic_1_network,
            gamma=gamma,
            lr=lr,
            tau=tau,
            clip_grad_norm=clip_grad_norm,
            device=device,
        )
        self._critic_1 = self._critic
        self._critic_2 = AddTargetNetwork(critic_2_network, device=device)
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip

        self._env_action_scale = env_action_scale
        self._env_action_min = env_action_min
        self._env_action_max = env_action_max
        self._optim_critic = torch.optim.Adam(
            params=list(self._critic_1.net.parameters())
            + list(self._critic_2.net.parameters()),
            lr=lr,
        )
        self._critic_update_frequency = critic_update_frequency
        self._policy_update_frequency = policy_update_frequency

        self._loss_actor_value = 0
        self._loss_critic_value = 0

    def _critic_update(self, batch: Transitions, aux: AuxiliaryMemoryData) -> float:
        """Runs a critic update

        Args:
            s_0 (Tensor): Observation before action
            a (Tensor): Agent action
            r (Tensor): Reward (s_0, a) -> (s_1)
            s_1 (Tensor): Observation after action
            d (Tensor): Dones

        Returns:
            float: critic loss
        """
        # Improvement 1: Clipped Double-Q Learning
        # ====================
        x_1 = self._critic_1.net.q(batch.s_0, batch.a).squeeze()
        x_2 = self._critic_2.net.q(batch.s_0, batch.a).squeeze()
        with torch.no_grad():
            a_1 = self._actor.target.a(batch.s_1)
            # Improvement 2: Target policy smoothing
            # ====================
            a_1 += (
                torch.clamp(
                    torch.normal(0, self.target_noise, a_1.shape, device=self._device),
                    min=-self.target_noise_clip,
                    max=self.target_noise_clip,
                )
                * self._env_action_scale
            )
            a_1 = torch.clamp(a_1, min=self._env_action_min, max=self._env_action_max)
            # ====================
            target_max_1 = self._critic_1.target.q(batch.s_1, a_1).squeeze()
            target_max_2 = self._critic_2.target.q(batch.s_1, a_1).squeeze()
            y = (
                batch.r
                + (~batch.d) * torch.minimum(target_max_1, target_max_2) * self._gamma
            )
        loss_critic = torch.mean((aux.weights * (y - x_1)) ** 2) + torch.mean(
            (aux.weights * (y - x_2)) ** 2
        )
        loss_value = loss_critic.item()
        self._optim_critic.zero_grad()
        loss_critic.backward()
        if not self._clip_grad_norm is None:
            nn.utils.clip_grad_norm_(
                self._critic_1.net.parameters(), self._clip_grad_norm
            )
            nn.utils.clip_grad_norm_(
                self._critic_2.net.parameters(), self._clip_grad_norm
            )
        self._optim_critic.step()
        # ====================

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
        if step % self._critic_update_frequency == 0:
            self._loss_critic_value = self._critic_update(batch, aux)
        if step % self._policy_update_frequency == 0:
            self._loss_actor_value = self._actor_update(batch.s_0, aux.weights)
            self._critic_1.update_target_network(tau=self._tau)
            self._critic_2.update_target_network(tau=self._tau)
            self._actor.update_target_network(tau=self._tau)

        return self._loss_critic_value, self._loss_actor_value

    def get_models(self) -> list[tuple[nn.Module, str]]:
        return [
            (self._actor.net, "actor"),
            (self._critic_1.net, "critic"),
            (self._critic_2.net, "critic_2"),
        ]
