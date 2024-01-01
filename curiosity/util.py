from typing import Optional
import random

import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers.autoreset import AutoResetWrapper 
from gymnasium.spaces import Box, Discrete
import numpy as np
import torch
import torch.nn as nn

from curiosity.nn import ClassicalBoxActor
from curiosity.intrinsic.intrinsic import IntrinsicReward, NoIntrinsicReward
from curiosity.intrinsic.icm import IntrinsicCuriosityModule
from curiosity.intrinsic.rnd import RandomNetworkDistillation
from curiosity.rl.ddpg import DeepDeterministicPolicyGradient
from curiosity.rl.td3 import TwinDelayedDeepDeterministicPolicyGradient

def build_env(environment_configuration):
    env = AutoResetWrapper(gym.make(environment_configuration.name, render_mode="rgb_array"))
    env.reset()
    return env

def build_rl(env, algorithm_configuration, device: str) -> IntrinsicReward:
    if algorithm_configuration.type == "ddpg":
        return DeepDeterministicPolicyGradient(
            build_actor(env, **algorithm_configuration.actor),
            build_critic(env, **algorithm_configuration.critic),
            device=device,
            **algorithm_configuration
        )
    elif algorithm_configuration.type =="td3":
        env_action_scale = torch.tensor(env.action_space.high-env.action_space.low, device=device) / 2.0
        env_action_min = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
        env_action_max = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)
        return TwinDelayedDeepDeterministicPolicyGradient(
            build_actor(env, **algorithm_configuration.actor),
            build_critic(env, **algorithm_configuration.critic),
            build_critic(env, **algorithm_configuration.critic),
            device=device,
            env_action_scale=env_action_scale,
            env_action_min=env_action_min,
            env_action_max=env_action_max,
            **algorithm_configuration
        )
    elif algorithm_configuration.type == "dqn":
        raise NotImplementedError()
    raise ValueError("Reinforcement learning algorithm type not valid")

def build_actor(env: Env, features: int = 128) -> nn.Module:
    """Builds an actor for gym environment

    Args:
        env (Env): Training gym environment
        feature (int): Width of each NN layer
 
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
        result = ClassicalBoxActor(env, features=features)
    else:
        raise NotImplementedError("Pixel space is WIP")

    return result

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
        result =  nn.Sequential(
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
        result =  AtariNetwork()

    return result


def build_intrinsic(env, intrinsic_configuration, device: str) -> IntrinsicReward:
    if intrinsic_configuration.type == "icm":
        return build_icm(env=env, device=device, **intrinsic_configuration)
    elif intrinsic_configuration.type =="rnd":
        return build_rnd(env=env, device=device, **intrinsic_configuration)
    return NoIntrinsicReward()

def build_icm(env: Env, encoding_size: int, device: str, clip_grad_norm: Optional[float] = 1, **kwargs):
    """ Builds a default intrinsic curiosity module
    Args:
        env (Env): gym environment
        encoding_size (int): Number of features for the encoding
        device (str, optional): Tensor device. Defaults to "cpu".
    Returns:
        _type_: ICM
    """
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    feature_net = nn.Sequential(
        nn.Linear(in_features=obs_size, out_features=encoding_size),
        nn.LeakyReLU(),
        nn.Linear(in_features=encoding_size, out_features=encoding_size),
        nn.LeakyReLU()
    ).to(device=device)
    forward_head = nn.Sequential(
        nn.Linear(in_features=encoding_size+action_size, out_features=encoding_size),
        nn.LeakyReLU(),
        nn.Linear(in_features=encoding_size, out_features=encoding_size)
    ).to(device=device)
    inverse_head = nn.Sequential(
        nn.Linear(in_features=encoding_size * 2, out_features=encoding_size),
        nn.LeakyReLU(),
        nn.Linear(in_features=encoding_size, out_features=action_size)
    ).to(device=device)

    result =  IntrinsicCuriosityModule(feature_net, forward_head, inverse_head, discrete_action_space=False, **kwargs)
    return result

def build_rnd(env: Env, encoding_size: int, device:str, lr: float = 1e-3, **kwargs):
    obs_size = env.observation_space.shape[0]
    target_net = nn.Sequential(
        nn.Linear(in_features=obs_size, out_features=encoding_size),
        nn.LeakyReLU(),
        nn.Linear(in_features=encoding_size, out_features=encoding_size),
        nn.LeakyReLU()
    ).to(device=device)
    predictor_net = nn.Sequential(
        nn.Linear(in_features=obs_size, out_features=encoding_size),
        nn.LeakyReLU(),
        nn.Linear(in_features=encoding_size, out_features=encoding_size),
        nn.LeakyReLU()
    ).to(device=device)
    return RandomNetworkDistillation(target_net, predictor_net, lr)

def global_seed(seed: int, *envs):
    """ Utility to help set determinism

    Args:
        seed (int): seed for rng generation
        envs (Env): all environments
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    for env in envs:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return np.random.default_rng(seed)