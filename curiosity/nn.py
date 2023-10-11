import copy
from typing import Union

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import torch
import torch.nn as nn

class AddTargetNetwork(nn.Module):
    """ Wraps an extra target network for delayed updates
    """
    def __init__(self, net: nn.Module, device = "cpu"):
        super().__init__()
        self.net = net
        self.target = copy.deepcopy(net)
        self.net = self.net.to(device)
        self.target = self.target.to(device)
    
    def forward(self, x, *args, **kwargs):
        return self.net(x, *args, **kwargs)

    def update_target_network(self, tau: float = 1) -> None:
        """ Updates target network to follow latest parameters

        Args:
            tau (float): Weighting factor. Defaults to 1.
        """
        assert (0 <= tau <= 1), f"Weighting {tau} is out of range"
        for target_param, net_param in zip(self.target.parameters(), self.net.parameters()):
            target_param.data.copy_(net_param.data * tau + target_param.data*(1-tau))

class GaussianActor(nn.Module):
    """ Implements a policy module with normal noise applied, for continuous environments.
    """
    def __init__(self, mean_network: nn.Module, std: Union[float, torch.Tensor, nn.Module]):
        """ Initialises the gaussian actor

        Args:
            mean_network (nn.Module): NN from observations to the mean.
            std_network (nn.Module): NN from observations to std.
        """
        super().__init__()
        self.mean = mean_network
        self.std = std

    def forward(self, x: torch.Tensor, noise: bool = False) -> torch.Tensor:
        """ Calculates the action to take

        Args:
            x (torch.Tensor): obs
            noise (bool, optional): Toggle to add gaussian noise.
                For example, set to false when evaluating.
                Defaults to False.

        Returns:
            torch.Tensor: action
        """
        action = self.mean(x)
        if noise:
            std = self.std
            if isinstance(std, nn.Module):
                std = self.std(x)
            action = torch.normal(action, std)
        return action

class ClassicalGaussianActor(GaussianActor):
    """ An actor for continuous low-dimensionality gym environments
    """
    def __init__(self, env: Env, features=128, exploration_factor=0.1):
        mean = nn.Sequential(
            nn.LazyLinear(out_features=features),
            nn.Tanh(),
            nn.Linear(in_features=features, out_features=features),
            nn.Tanh(),
            nn.Linear(in_features=features, out_features=env.action_space.shape[-1]),
            nn.Tanh()
        )
        super().__init__(mean, exploration_factor)
        self.register_buffer("scale", torch.tensor(env.action_space.high-env.action_space.low) / 2.0)
        self.register_buffer("bias", torch.tensor(env.action_space.high+env.action_space.low) / 2.0)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs) * self.scale + self.bias

def build_critic(env: Env, features: int):
    """ Build a default critic network for env on a single-agent environment
    """
    # Observation space
    # Action space
    discrete = isinstance(env.action_space, Discrete)
    if not discrete and len(env.action_space.shape) != 1:
        raise ValueError(f"Action space {env.action_space} is not supported")

    # Observation space
    if not isinstance(env.observation_space, Box):
        raise ValueError(f"Observation space {env.observation_space} is not supported")
    pixels = len(env.observation_space.shape) != 1

    if not pixels:
        return nn.Sequential(
            nn.Linear(in_features=env.observation_space.shape[-1] + (0 if discrete else len(env.action_space.shape)), out_features=features),
            nn.Tanh(),
            nn.Linear(in_features=features, out_features=features),
            nn.Tanh(),
            nn.Linear(in_features=features, out_features=env.action_space.n if discrete else 1)
        )
    else:
        # Architecture from OpenAI baselines
        class AtariNetwork(nn.Module):
            def __init__(self, atari_grayscale: bool = True):
                super().__init__()
                self.grayscale = atari_grayscale
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
                    nn.ReLU(),
                )
                self.flatten = nn.Flatten()
                self.linear = nn.LazyLinear(env.action_space.n)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.grayscale:
                    x = x.unsqueeze(-1)
                    x = torch.transpose(x, -1, -3)
                if len(x.shape) == 3:
                    x = x.unsqueeze(0)
                result = self.conv(x)
                result = self.flatten(result)
                result = self.linear(result)
                result = result.squeeze()
                return result

        return AtariNetwork()
# TODO(mark)
# - Whats a good method for sharing layers between actor and critic. Is it even neccessary?
