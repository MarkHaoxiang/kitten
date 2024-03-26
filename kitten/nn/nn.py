from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium import Env
import torch
from torch import Tensor
import torch.nn as nn

from kitten.policy.interface import PolicyFn


class Value(ABC, nn.Module):
    """Analogous to V networks"""

    @abstractmethod
    def v(self, s: Tensor) -> Tensor:
        raise NotImplementedError


class HasValue(ABC):
    """Reinforcement Learning Algorithm with a Value Module"""

    @property
    @abstractmethod
    def value(self) -> Value:
        raise NotImplementedError


class Critic(ABC, nn.Module):
    """Analogous to the Q Network"""

    @abstractmethod
    def q(self, s: Tensor, a: Tensor) -> Tensor:
        """Calculates the value of taking an action under a state

        Args:
            s (torch.Tensor): State.
            a (torch.Tensor): Action.

        Returns:
            torch.Tensor: Value.
        """
        raise NotImplementedError


class HasCritic(ABC):
    """Reinforcement Learning Algorithm with a Critic Module"""

    @property
    @abstractmethod
    def critic(self) -> Critic:
        raise NotImplementedError


class CriticPolicyPair(Critic, Value):
    def __init__(self, critic: Critic, policy_fn: PolicyFn) -> None:
        super().__init__()
        self._critic = critic
        self._policy_fn = policy_fn

    def q(self, s: Tensor, a: Tensor) -> Tensor:
        return self._critic.q(s, a)

    def v(self, s: Tensor) -> Tensor:
        return self._critic.q(s, self._policy_fn(s))


class Actor(ABC, nn.Module):
    """Analogous to Pi Network"""

    @abstractmethod
    def a(self, s: Tensor) -> Tensor:
        """Calculates the desired action

        Args:
            s (Tensor): State.

        Returns:
            Tensor: Action.
        """
        raise NotImplementedError

    def to_policy_fn(self) -> PolicyFn:
        return lambda s: self(s)


class ClassicalBoxCritic(Critic):
    """Critic for continuous low-dimensionality gym environments"""

    def __init__(
        self,
        env: Env[NDArray[Any], NDArray[Any]],
        net: nn.Module | None = None,
        features: int = 128,
    ):
        super().__init__()
        assert isinstance(env.action_space, gym.spaces.Box)
        assert isinstance(env.observation_space, gym.spaces.Box)
        if net is not None:
            self.net = net
        else:
            self.net = nn.Sequential(
                nn.Linear(
                    in_features=env.observation_space.shape[-1]
                    + env.action_space.shape[0],
                    out_features=features,
                ),
                nn.LeakyReLU(),
                nn.Linear(in_features=features, out_features=features),
                nn.LeakyReLU(),
                nn.Linear(in_features=features, out_features=1),
            )

    def q(self, s: Tensor, a: Tensor) -> Tensor:
        combined = torch.cat((s, a), dim=-1)
        return self(combined)

    def forward(self, x: Tensor):
        return self.net(x)


class ClassicalDiscreteCritic(Critic):
    """Critic for discrete low-dimensionality gym environments"""

    def __init__(
        self,
        env: Env[NDArray[Any], np.int64],
        net: nn.Module | None = None,
        features: int = 128,
    ):
        super().__init__()
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert isinstance(env.observation_space, gym.spaces.Box)

        if net is not None:
            self.net = net
        else:
            self.net = nn.Sequential(
                nn.Linear(
                    in_features=env.observation_space.shape[-1], out_features=features
                ),
                nn.LeakyReLU(),
                nn.Linear(in_features=features, out_features=features),
                nn.LeakyReLU(),
                nn.Linear(in_features=features, out_features=int(env.action_space.n)),
            )

    def q(self, s: Tensor, a: Tensor | None = None) -> Tensor:
        if a is None:
            return self(s)
        else:
            return self(s)[a]

    def forward(self, x: Tensor):
        return self.net(x)


class ClassicalBoxActor(Actor):
    """Actor for continuous low-dimensionality gym environments"""

    def __init__(
        self,
        env: Env[NDArray[Any], NDArray[Any]],
        net: nn.Module | None = None,
        features: int = 128,
    ):
        super().__init__()
        assert isinstance(env.action_space, gym.spaces.Box)
        assert isinstance(env.observation_space, gym.spaces.Box)
        if net is None:
            self.net: nn.Module = nn.Sequential(
                nn.Linear(
                    in_features=env.observation_space.shape[0], out_features=features
                ),
                nn.LeakyReLU(),
                nn.Linear(in_features=features, out_features=features),
                nn.LeakyReLU(),
                nn.Linear(
                    in_features=features, out_features=env.action_space.shape[-1]
                ),
                nn.Tanh(),
            )
        else:
            self.net = net
        self.register_buffer(
            "scale", torch.tensor(env.action_space.high - env.action_space.low) / 2.0
        )
        self.register_buffer(
            "bias", torch.tensor(env.action_space.high + env.action_space.low) / 2.0
        )

    def a(self, s: Tensor) -> Tensor:
        return self(s)

    def forward(self, x: Tensor):
        return self.net(x) * self.scale + self.bias
