from abc import ABC, abstractmethod
import copy
from typing import Callable

import gymnasium as gym
from gymnasium import Env
import torch
from torch import Tensor
import torch.nn as nn


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


class Value(ABC, nn.Module):
    """Analogous to V networks"""

    @abstractmethod
    def v(self, s: Tensor) -> Tensor:
        raise NotImplementedError


# TODO: Generator for Value from Critic and Actor
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

    def to_policy_fn(self) -> Callable:
        return lambda s: self(s)

class ClassicalBoxCritic(Critic):
    """Critic for continuous low-dimensionality gym environments"""

    def __init__(self, env: Env, net: nn.Module | None = None, features: int = 128):
        super().__init__()
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("Unsupported Environment")
        if net is None:
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
        else:
            self.net = net

    def q(self, s: Tensor, a: Tensor) -> Tensor:
        combined = torch.cat((s, a), dim=-1)
        return self(combined)

    def forward(self, x: Tensor):
        return self.net(x)


class ClassicalDiscreteCritic(Critic):
    """Critic for discrete low-dimensionality gym environments"""

    def __init__(self, env: Env, net: nn.Module | None = None, features: int = 128):
        super().__init__()
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("Unsupported Environment")
        if net is None:
            self.net = nn.Sequential(
                nn.Linear(
                    in_features=env.observation_space.shape[-1], out_features=features
                ),
                nn.LeakyReLU(),
                nn.Linear(in_features=features, out_features=features),
                nn.LeakyReLU(),
                nn.Linear(in_features=features, out_features=env.action_space.n),
            )
        else:
            self.net = net

    def q(self, s: Tensor, a: Tensor | None = None) -> Tensor:
        if a is None:
            return self(s)
        else:
            return self(s)[a]

    def forward(self, x: Tensor):
        return self.net(x)


class ClassicalBoxActor(Actor):
    """Actor for continuous low-dimensionality gym environments"""

    def __init__(self, env: Env, net: nn.Module | None = None, features: int = 128):
        super().__init__()
        if net is None:
            self.net = nn.Sequential(
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
