from abc import ABC, abstractmethod
import copy
from typing import Callable, Optional

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


class AddTargetNetwork(nn.Module):
    """Wraps an extra target network for delayed updates"""

    def __init__(self, net: nn.Module, device="cpu"):
        super().__init__()
        self.net = net
        self.target = copy.deepcopy(net)
        self.net = self.net.to(device)
        self.target = self.target.to(device)

    def forward(self, x, *args, **kwargs):
        return self.net(x, *args, **kwargs)

    def update_target_network(self, tau: float = 1) -> None:
        """Updates target network to follow latest parameters

        Args:
            tau (float): Weighting factor. Defaults to 1.
        """
        assert 0 <= tau <= 1, f"Weighting {tau} is out of range"
        for target_param, net_param in zip(
            self.target.parameters(), self.net.parameters()
        ):
            target_param.data.copy_(
                net_param.data * tau + target_param.data * (1 - tau)
            )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return self.net.__getattribute__(name)


class ClassicalBoxCritic(Critic):
    """Critic for continuous low-dimensionality gym environments"""

    def __init__(self, env: Env, net: Optional[nn.Module] = None, features: int = 128):
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

    def __init__(self, env: Env, net: Optional[nn.Module] = None, features: int = 128):
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

    def q(self, s: Tensor, a: Optional[Tensor] = None) -> Tensor:
        if a is None:
            return self(s)
        else:
            return self(s)[a]

    def forward(self, x: Tensor):
        return self.net(x)


class ClassicalBoxActor(Actor):
    """Actor for continuous low-dimensionality gym environments"""

    def __init__(self, env: Env, net: Optional[nn.Module] = None, features: int = 128):
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
