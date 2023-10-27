from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import torch
import torch.nn as nn

from curiosity.nn import ClassicalGaussianActor

def build_actor(env: Env, features: int = 128, exploration_factor: float=0.1) -> nn.Module:
    """Builds an actor for gym environment

    Args:
        env (Env): Training gym environment
        feature (int): Width of each NN layer
        exploration_factor (float): Exploration noise to add
    
    Returns:
        nn.Module: Actor NNs
    """
    # Action space
    discrete = isinstance(env.action_space, Discrete)
    if not discrete and len(env.action_space.shape) != 1:
        raise ValueError(f"Action space {env.action_space} is not supported")

    # Observation space
    if not isinstance(env.observation_space, Box):
        raise ValueError(f"Observation space {env.observation_space} is not supported")
    pixels = len(env.observation_space.shape) != 1

    if not pixels:
        if discrete:
            raise NotImplementedError("Discrete action space is a WiP")
        return ClassicalGaussianActor(
            env,
            features=features,
            exploration_factor=exploration_factor
        )
    else:
        raise NotImplementedError("Pixel space is WIP")

def build_critic(env: Env, features: int) -> nn.Module:
    """Builds a critic for gym environment

    Args:
        env (Env): Training gym environment
        features (int): Width of each NN layer

    Returns:
        nn.Module: Critic NN
    """
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
            nn.Linear(in_features=env.observation_space.shape[-1] + (0 if discrete else env.action_space.shape[0]), out_features=features),
            nn.LeakyReLU(),
            nn.Linear(in_features=features, out_features=features),
            nn.LeakyReLU(),
            nn.Linear(in_features=features, out_features=env.action_space.n if discrete else 1)
        )
    else:
        # Architecture from OpenAI baselines
        if not discrete:
            raise NotImplementedError("Pixel space with continuous actions is not yet implemented.")
        class AtariNetwork(nn.Module):
            def __init__(self, atari_grayscale: bool = True):
                super().__init__()
                self.grayscale = atari_grayscale
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=4),
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
                    nn.LeakyReLU(),
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