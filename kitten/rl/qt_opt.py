import gymnasium as gym
from numpy import ndarray
import torch
from torch import Tensor
import torch.nn as nn

from kitten.experience import AuxiliaryMemoryData, Transitions
from kitten.rl import Algorithm, HasCritic
from kitten.nn import Critic, AddTargetNetwork


def cross_entropy_method(
    s_0: torch.Tensor,
    critic_network: Critic,
    action_space: gym.spaces.Box,
    n: int = 64,
    m: int = 6,
    n_iterations: int = 2,
    device: str = "cpu",
) -> torch.Tensor:
    """Sampling method to find action for a Q function on continuous spaces

    Args:
        critic_network (torch.nn): Critic.
        action_space (gym.spaces.Box): Action space
        n (int, optional): Number of values each iteration.
        m (int, optional): Best values each iteration
        n_iterations (int, optional): Number of convergence iterations.
        device (str, optional): Device of critic and output.

    Returns:
        torch.Tensor: Action
    """
    if n_iterations <= 0:
        raise ValueError(
            "Number of iterations must be at least 1, currently {n_iterations}"
        )

    # For clipping
    env_action_min = torch.tensor(action_space.low, dtype=torch.float32, device=device)
    env_action_max = torch.tensor(action_space.high, dtype=torch.float32, device=device)

    # Reshape
    s_0 = s_0.unsqueeze(1).expand((s_0.shape[0], n, *s_0.shape[1:]))

    # Run cem
    sampled_actions = torch.rand((s_0.shape[0], n, *action_space.shape), device=device)
    sampled_actions = (
        sampled_actions * (env_action_max - env_action_min)[None, None, :]
        + env_action_min[None, None, :]
    )
    batch_indices = torch.arange(sampled_actions.shape[0]).view(-1, 1).expand(-1, m)

    for _ in range(n_iterations):
        q = critic_network.q(s_0, sampled_actions)
        indices = torch.topk(q, m, dim=1).indices.squeeze()
        best_actions = sampled_actions[batch_indices, indices, :]
        mu, std = best_actions.mean(dim=1), best_actions.std(dim=1)
        sampled_actions = (
            torch.randn_like(sampled_actions) * std[:, None, :] + mu[:, None, :]
        )
        sampled_actions = torch.clamp(
            sampled_actions, min=env_action_min, max=env_action_max
        )

    return mu


class QTOpt(Algorithm, HasCritic):
    """Q-learning for continuous actions with Cross-Entropy Maximisation

    With clipped Double DQN

    Kalashnikov et al. QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation
    """

    def __init__(
        self,
        critic_1_network: Critic,
        critic_2_network: Critic,
        obs_space: gym.spaces.Box,  # This is a hack #TODO: How do we pass batch sizes around?
        action_space: gym.spaces.Box,
        gamma: float = 0.99,
        lr: float = 1e-3,
        tau: float = 0.005,
        cem_n: int = 64,
        cem_m: int = 6,
        cem_n_iterations: int = 2,
        clip_grad_norm: float | None = 1.0,
        update_frequency: int = 1.0,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()

        self.device = device

        self._critic_1 = AddTargetNetwork(critic_1_network, device=device)
        self._critic_2 = AddTargetNetwork(critic_2_network, device=device)
        self._gamma = gamma
        self._tau = tau
        self._env_action_scale = (
            torch.tensor(action_space.high - action_space.low, device=device) / 2.0
        )
        self._env_action_min = torch.tensor(
            action_space.low, dtype=torch.float32, device=device
        )
        self._env_action_max = torch.tensor(
            action_space.low, dtype=torch.float32, device=device
        )
        self._obs_space = obs_space
        self._action_space = action_space
        self._clip_grad_norm = clip_grad_norm
        self._update_frequency = update_frequency
        self._n = cem_n
        self._m = cem_m
        self._n_iterations = cem_n_iterations

        self._optim_critic = torch.optim.Adam(
            params=list(self._critic_1.net.parameters())
            + list(self._critic_2.net.parameters()),
            lr=lr,
        )

        self.loss_critic_value = 0

    def policy_fn(self, s: Tensor | ndarray, critic: Critic | None = None) -> Tensor:
        if isinstance(s, ndarray):
            s = torch.tensor(s, device=self.device)
        if critic is None:
            critic = self._critic_1
        squeeze = False
        if self._obs_space.shape == s.shape:
            squeeze = True
            s = s.unsqueeze(0)
        result = cross_entropy_method(
            s_0=s,
            critic_network=critic,
            action_space=self._action_space,
            n=self._n,
            m=self._m,
            n_iterations=self._n_iterations,
            device=self.device,
        )
        if squeeze:
            result = result.squeeze()
        return result

    def td_error(self, s_0: Tensor, a: Tensor, r: Tensor, s_1: Tensor, d: Tensor):
        """Returns TD difference for a transition"""
        x = self._critic_1.q(s_0, a).squeeze()
        with torch.no_grad():
            a_1 = self.policy_fn(s_1)
            target_max = (~d) * self.critic.target.q(s_1, a_1).squeeze()
            y = r + target_max * self._gamma
        return y - x

    def _critic_update(self, batch: Transitions, aux: AuxiliaryMemoryData):
        x_1 = self._critic_1.q(batch.s_0, batch.a).squeeze()
        x_2 = self._critic_2.q(batch.s_0, batch.a).squeeze()
        with torch.no_grad():
            a_1 = self.policy_fn(batch.s_1, critic=self._critic_1.target)
            a_2 = self.policy_fn(batch.s_1, critic=self._critic_2.target)
            target_max_1 = self._critic_1.target.q(batch.s_1, a_1).squeeze()
            target_max_2 = self._critic_2.target.q(batch.s_1, a_2).squeeze()
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

        return loss_value

    def update(self, batch: Transitions, aux: AuxiliaryMemoryData, step: int):
        if step % self._update_frequency == 0:
            self.loss_critic_value = self._critic_update(batch, aux)
            self._critic_1.update_target_network(tau=self._tau)
            self._critic_2.update_target_network(tau=self._tau)
        return self.loss_critic_value

    def get_log(self):
        return {
            "critic_loss": self.loss_critic_value,
        }

    def get_models(self) -> list[tuple[nn.Module, str]]:
        return [(self._critic_1.net, "critic"), (self._critic_2.net, "critic_2")]

    @property
    def critic(self) -> Critic:
        return self._critic_1
