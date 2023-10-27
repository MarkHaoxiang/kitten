from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn

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
                 target_update_frequency: int = 1,
                 device: str = "cpu"):
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

        self._calc_critic_loss = nn.MSELoss()
        self._gamma = gamma
        self._optim_actor = torch.optim.Adam(params=self.actor.net.parameters(), lr=lr)
        self._optim_critic = torch.optim.Adam(params=self.critic.net.parameters(), lr=lr)
        self._tau = tau

    def _critic_update(self, s_0: Tensor, a: Tensor, r: Tensor, s_1: Tensor, d: Tensor) -> float:
        """ Runs a critic update

        Args:
            s_0 (Tensor): Observation before action
            a (Tensor): Agent action
            r (Tensor): Reward (s_0, a) -> (s_1)
            s_1 (Tensor): Observation after action
            d (Tensor): Dones

        Returns:
            float: critic loss
        """
        x = self.critic(torch.cat((s_0, a), 1)).squeeze()
        with torch.no_grad():
            target_max = (~d) * self.critic.target(torch.cat((s_1, self.actor.target(s_1)), 1)).squeeze()
            y = (r + target_max * self._gamma)
        
        loss = self._calc_critic_loss(y, x)
        loss_value = loss.item()
        self._optim_critic.zero_grad()
        loss.backward()
        self._optim_critic.step()

        return loss_value

    def _actor_update(self, s_0: Tensor) -> float:
        """ Runs a critic update

        Args:
            s_0 (Tensor): Observation before action
            a (Tensor): Agent action
            r (Tensor): Reward (s_0, a) -> (s_1)
            s_1 (Tensor): Observation after action
            d (Tensor): Dones

        Returns:
            float: actor loss
        """
        desired_action = self.actor(s_0)
        loss = -self.critic(torch.cat((s_0, desired_action), 1)).mean()
        loss_value = loss.item()
        self._optim_actor.zero_grad()
        loss.backward()
        self._optim_actor.step()

        return loss_value
    
    def update(self, batch: Tuple) -> Tuple[float, float]:
        """ Runs a DDPG update step

        Args:
            batch (Tuple): Batch of training transitions from the replay buffer

        Returns:
            Tuple[float, float]: Critic loss and actor loss
        """
        loss_critic_value = self._critic_update(*batch)
        loss_actor_value = self._actor_update(batch[0])
        self.critic.update_target_network(tau=self._tau)
        self.actor.update_target_network(tau=self._tau)

        return loss_critic_value, loss_actor_value

class TwinDelayedDeepDeterministicPolicyGradient(DeepDeterministicPolicyGradient):
    """ TD3
    """
    def __init__(self,
                 actor: nn.Module,
                 critic_1: nn.Module,
                 critic_2: nn.Module,
                 env_action_scale,
                 env_action_min,
                 env_action_max,
                 gamma: float = 0.99,
                 lr: float=1e-3,
                 tau: float = 0.005,
                 target_noise: float = 0.1,
                 target_noise_clip: float = 0.2,
                 device: str = "cpu"):
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
            actor=actor,
            critic=critic_1,
            gamma=gamma,
            lr=lr,
            tau=tau,
            device=device
        )
        self.critic_1 = self.critic
        self.critic_2 = AddTargetNetwork(critic_2, device=device)
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip

        self._env_action_scale = env_action_scale
        self._env_action_min = env_action_min
        self._env_action_max = env_action_max
        self._optim_critic = torch.optim.Adam(
            params=list(self.critic_1.net.parameters()) + list(self.critic_2.net.parameters()),
            lr=lr
        )

    def _critic_update(self, s_0: Tensor, a: Tensor, r: Tensor, s_1: Tensor, d: Tensor) -> float:
        """ Runs a critic update

        Args:
            s_0 (Tensor): Observation before action
            a (Tensor): Agent action
            r (Tensor): Reward (s_0, a) -> (s_1)
            s_1 (Tensor): Observation after action
            d (Tensor): Dones

        Returns:
            float: critic loss
        """
        s_0_a = torch.cat((s_0, a), 1)
        # Improvement 1: Clipped Double-Q Learning
        # ==================== 
        x_1 = self.critic_1(s_0_a).squeeze()
        x_2 = self.critic_2(s_0_a).squeeze()
        with torch.no_grad():
            a_1 = self.actor.target(s_1)
            # Improvement 2: Target policy smoothing
            # ==================== 
            a_1 += torch.clamp(
                torch.normal(0, self.target_noise, a_1.shape, device=self.device),
                min = -self.target_noise_clip,
                max = self.target_noise_clip,
            ) * self._env_action_scale
            a_1 = torch.clamp(
                a_1,
                min=self._env_action_min,
                max=self._env_action_max
            )
            s_1_a_1 = torch.cat((s_1, a_1), 1)
            # ==================== 
            target_max_1 = self.critic_1.target(s_1_a_1).squeeze()
            target_max_2 = self.critic_2.target(s_1_a_1).squeeze()
            y = r + (~d) * torch.minimum(target_max_1, target_max_2) * self._gamma

        loss_critic = self._calc_critic_loss(y, x_1) + self._calc_critic_loss(y, x_2)
        loss_value = loss_critic.item()
        self._optim_critic.zero_grad()
        loss_critic.backward()
        self._optim_critic.step()
        # ==================== 

        return loss_value
    
    def update(self, *args, **kwargs):
        raise AttributeError("Please use critic_update and policy_update")
    
    def critic_update(self, batch: Tuple) -> float:
        """ Runs a TD3 critic step

        Args:
            batch (Tuple): Batch of training transitions from the replay buffer

        Returns:
            float: Critic loss
        """
        loss_critic_value = self._critic_update(*batch)
        return loss_critic_value

    def actor_update(self, batch: Tuple) -> float:
        """ Runs a TD3 actor step

        Args:
            batch (Tuple): Batch of training transitions from the replay buffer

        Returns:
            float: Actor loss
        """
        loss_actor_value = self._actor_update(batch[0])
        self.critic_1.update_target_network(self._tau)
        self.critic_2.update_target_network(self._tau)
        self.actor.update_target_network(self._tau)
        return loss_actor_value