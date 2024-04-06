from typing import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import gymnasium as gym
from jaxtyping import Float, Bool

from kitten.common.rng import Generator
from kitten.experience import Transitions
from kitten.rl import Algorithm
from kitten.rl.qt_opt import cross_entropy_method
from kitten.experience import AuxiliaryMemoryData
from kitten.nn import Critic, AddTargetNetwork, HasCritic, HasValue, CriticPolicyPair, Ensemble


@dataclass
class ResetValueMixin:
    v_1: Float[Tensor, "..."]  # Value of taking a reset
    reset_value_mixin_select: Bool[Tensor, "..."]  # Which values to override
    reset_value_mixin_enable: bool = False  # Enable


# For now, uses Mixins.
# The ideal type would be to wait until intersections are implemented
# This would be equivalent to Rust-like traits
# https://github.com/python/typing/issues/213
@dataclass
class ResetValueOverloadAux(ResetValueMixin, AuxiliaryMemoryData):
    pass


class QTOptCats(Algorithm, HasCritic, HasValue):
    """QtOpt with Ensembles and other modifications"""

    def __init__(
        self,
        build_critic: Callable[[], Critic],
        obs_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        rng: Generator,
        ensemble_number: int = 5,
        gamma: float = 0.99,
        lr: float = 1e-3,
        tau: float = 0.005,
        cem_n: int = 64,
        cem_m: int = 6,
        cem_n_iterations: int = 2,
        clip_grad_norm: float | None = 1,
        update_frequency: int = 1,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()

        self.critics = Ensemble(
            lambda: AddTargetNetwork(build_critic(), device=device),
            n=ensemble_number
        )

        self.device = device

        self._ensemble_number = ensemble_number
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
        self._chosen_critic = self.critics.sample_network().net

        self._optim_critic = torch.optim.Adam(
            params=self.critics.parameters(), lr=lr
        )

        self.loss_critic_value = 0

    @property
    def critic(self):
        return self._chosen_critic

    @property
    def value(self):
        return CriticPolicyPair(self.critic, self.policy_fn)

    def mu_var(self, s: Tensor, a: Tensor | None = None):
        """Calculates expectation and epistemic uncertainty of a state-action pair"""
        with torch.no_grad():
            if a is None:
                a = self.policy_fn(s)
            q = np.array([critic.net.q(s, a).cpu() for critic in self.critics])
        mu, var = q.mean(axis=0), q.var(axis=0)
        return mu, var

    def policy_fn(self, s: Tensor | np.ndarray, critic: Critic | None = None) -> Tensor:
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, device=self.device)
        if critic is None:
            critic = self.critic
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

    def reset_critic(self):
        self._chosen_critic = self.critics.sample_network().net

    def _critic_update(self, batch: Transitions, aux: ResetValueOverloadAux):
        x = [critic.net.q(batch.s_0, batch.a).squeeze() for critic in self.critics]
        with torch.no_grad():
            # Implement a variant of Clipped Double-Q Learning
            # Randomly sample two networks
            c_1 = self.critics.sample_network()
            c_2 = self.critics.sample_network()
            a_1 = self.policy_fn(batch.s_1, critic=c_1.target)
            a_2 = self.policy_fn(batch.s_1, critic=c_2.target)
            target_max_1 = c_1.target.q(batch.s_1, a_1).squeeze()
            target_max_2 = c_2.target.q(batch.s_1, a_2).squeeze()
            y_1 = torch.minimum(target_max_1, target_max_2)
            if aux.reset_value_mixin_enable:
                y_1 = torch.where(
                    input=aux.v_1, other=y_1, condition=aux.reset_value_mixin_select
                )
            y = batch.r + (~batch.d) * y_1 * self._gamma

        losses = []
        for x_i in x:
            losses.append(torch.mean((aux.weights * (y - x_i)) ** 2))
        loss_critic = sum(losses)

        loss_value = loss_critic.item()
        self._optim_critic.zero_grad()
        loss_critic.backward()
        if not self._clip_grad_norm is None:
            for critic in self.critics:
                nn.utils.clip_grad_norm_(critic.net.parameters(), self._clip_grad_norm)
        self._optim_critic.step()

        return loss_value

    def update(self, batch: Transitions, aux: ResetValueOverloadAux, step: int):
        if step % self._update_frequency == 0:
            self.loss_critic_value = self._critic_update(batch, aux)
            for critic in self.critics:
                critic.update_target_network(tau=self._tau)
        return self.loss_critic_value

    def get_log(self):
        return {
            "critic_loss": self.loss_critic_value,
        }

    def get_models(self) -> list[tuple[nn.Module, str]]:
        return list(
            zip(self.critics, [f"critic_{i}" for i in range(len(self.critics))])
        )
