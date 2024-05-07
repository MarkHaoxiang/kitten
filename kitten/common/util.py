# type: ignore

from typing import Any

import hydra
from omegaconf import OmegaConf, DictConfig
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn

from kitten.nn import (
    Actor,
    Critic,
    ClassicalBoxActor,
    ClassicalBoxCritic,
    ClassicalDiscreteCritic,
)
from kitten.intrinsic import IntrinsicReward, NoIntrinsicReward
from kitten.intrinsic.icm import IntrinsicCuriosityModule
from kitten.intrinsic.rnd import RandomNetworkDistillation
from kitten.intrinsic.disagreement import Disagreement
from kitten.logging import KittenLogger, engine_registry
from kitten.rl import Algorithm
from kitten.rl.ddpg import DeepDeterministicPolicyGradient
from kitten.rl.td3 import TwinDelayedDeepDeterministicPolicyGradient
from kitten.rl.qt_opt import QTOpt
from kitten.rl.dqn import DQN
from kitten.policy import ColoredNoisePolicy, EpsilonGreedyPolicy, PolicyFn, Policy
from kitten.common.typing import Device

from .rng import Generator


def build_env(
    name: str,
    normalise_observation: bool = False,
    seed: int | Generator | np.random.Generator | None = None,
    **kwargs,
) -> gym.Env:
    """Utility to construct an environment

    Calls gym.make, applies extra wrappers and resets

    Args:
        name (str): Environment name within registry
        normalise_observation (bool, optional): Wraps NormalizeObservation.
            Use with online methods, and normalise with replay buffer otherwise. Defaults to False.

    Returns:
        gym.Env: Constructed environment
    """
    # Gym Make
    env = gym.make(name, render_mode="rgb_array", **kwargs)
    # Wrappers
    if normalise_observation:
        env = gym.wrappers.NormalizeObservation(env)
    # Reset and seeding
    if isinstance(seed, Generator):
        seed = int(seed.numpy.integers(65535))
    elif isinstance(seed, np.random.Generator):
        seed = int(seed.integers(65535))
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def build_rl(
    env: gym.Env[Any, Any], algorithm_configuration, device: Device
) -> Algorithm:
    """Utility to construct a RL algorithm"""
    if algorithm_configuration.type == "ddpg":
        assert isinstance(env.action_space, gym.spaces.Box)
        return DeepDeterministicPolicyGradient(
            build_actor(env, **algorithm_configuration.actor),
            build_critic(env, **algorithm_configuration.critic),
            device=device,
            **algorithm_configuration,
        )
    elif algorithm_configuration.type == "td3":
        assert isinstance(env.action_space, gym.spaces.Box)
        env_action_scale = (
            torch.tensor(env.action_space.high - env.action_space.low, device=device)
            / 2.0
        )
        env_action_min = torch.tensor(
            env.action_space.low, dtype=torch.float32, device=device
        )
        env_action_max = torch.tensor(
            env.action_space.high, dtype=torch.float32, device=device
        )
        return TwinDelayedDeepDeterministicPolicyGradient(
            build_actor(env, **algorithm_configuration.actor),
            build_critic(env, **algorithm_configuration.critic),
            build_critic(env, **algorithm_configuration.critic),
            device=device,
            env_action_scale=env_action_scale,
            env_action_min=env_action_min,
            env_action_max=env_action_max,
            **algorithm_configuration,
        )
    elif algorithm_configuration.type == "qt_opt":
        assert isinstance(env.action_space, gym.spaces.Box)
        assert isinstance(env.observation_space, gym.spaces.Box)
        return QTOpt(
            build_critic(env, **algorithm_configuration.critic),
            build_critic(env, **algorithm_configuration.critic),
            action_space=env.action_space,
            obs_space=env.observation_space,
            device=device,
            **algorithm_configuration,
        )

    elif algorithm_configuration.type == "dqn":
        assert isinstance(env.action_space, gym.spaces.Discrete)
        return DQN(
            build_critic(env, **algorithm_configuration.critic),
            device=device,
            **algorithm_configuration,
        )
    raise ValueError("Reinforcement learning algorithm type not valid")


def build_policy(
    policy_fn: PolicyFn,
    rng: Generator,
    env: Env,
    policy_cfg: dict[str, Any] | DictConfig,
    device: Device = "cpu",
) -> Policy:
    if isinstance(policy_cfg, DictConfig):
        policy_cfg = OmegaConf.to_container(policy_cfg)

    t = policy_cfg.pop("type")
    match t:
        case "colour":
            if "episode_length" not in policy_cfg:
                policy_cfg["episode_length"] = env.spec.max_episode_steps
            return ColoredNoisePolicy(
                fn=policy_fn,
                action_space=env.action_space,
                rng=rng,
                device=device,
                **policy_cfg,
            )
        case "e_greedy":
            return EpsilonGreedyPolicy(
                fn=policy_fn,
                action_space=env.action_space,
                rng=rng,
                device=device,
                **policy_cfg,
            )
        case _:
            raise ValueError("Unknown policy type")


def build_actor(env: Env, features: int = 128) -> Actor:
    """Builds an actor for gym environment

    Args:
        env (Env): Training gym environment
        feature (int): Width of each NN layer

    Returns:
        nn.Module: Actor NNs
    """
    # Action space
    discrete = isinstance(env.action_space, Discrete)
    if not discrete:
        assert isinstance(env.action_space, gym.spaces.Box)
        if len(env.action_space.shape) != 1:
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


def build_critic(env: Env, features: int = 128) -> Critic:
    """Builds a critic for gym environment

    Args:
        env (Env): Training gym environment
        features (int): Width of each NN layer

    Returns:
        nn.Module: Critic NN
    """
    # Action space
    discrete = isinstance(env.action_space, Discrete)
    if not discrete:
        assert isinstance(env.action_space, gym.spaces.Box)
        if len(env.action_space.shape) != 1:
            raise ValueError(f"Action space {env.action_space} is not supported")

    # Observation space
    if not isinstance(env.observation_space, Box):
        raise ValueError(f"Observation space {env.observation_space} is not supported")
    pixels = len(env.observation_space.shape) != 1

    if not pixels:
        if discrete:
            # TODO: Move into new build_value
            return ClassicalDiscreteCritic(env, features=features)
        else:
            return ClassicalBoxCritic(env, features=features)
    else:
        # Architecture from OpenAI baselines
        # Deprecated
        raise NotImplementedError
        """if not discrete:
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
        result =  AtariNetwork()"""


def build_intrinsic(
    env, intrinsic_configuration, device: Device = "cpu"
) -> IntrinsicReward:
    if intrinsic_configuration.type == "icm":
        return build_icm(env=env, device=device, **intrinsic_configuration)
    elif intrinsic_configuration.type == "rnd":
        return build_rnd(env=env, device=device, **intrinsic_configuration)
    elif intrinsic_configuration.type == "disagreement":
        return build_disagreement(env=env, device=device, **intrinsic_configuration)
    return NoIntrinsicReward()


def build_icm(
    env: Env[NDArray[Any], NDArray[Any]],
    encoding_size: int,
    device: Device,
    clip_grad_norm: float | None = 1.0,
    **kwargs,
):
    """Builds a default intrinsic curiosity module
    Args:
        env (Env): gym environment
        encoding_size (int): Number of features for the encoding
        device (str, optional): Tensor device. Defaults to "cpu".
    Returns:
        _type_: ICM
    """
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    feature_net = nn.Sequential(
        nn.Linear(in_features=obs_size, out_features=encoding_size),
        nn.LeakyReLU(),
        nn.Linear(in_features=encoding_size, out_features=encoding_size),
        nn.LeakyReLU(),
    ).to(device=device)
    forward_head = nn.Sequential(
        nn.Linear(in_features=encoding_size + action_size, out_features=encoding_size),
        nn.LeakyReLU(),
        nn.Linear(in_features=encoding_size, out_features=encoding_size),
    ).to(device=device)
    inverse_head = nn.Sequential(
        nn.Linear(in_features=encoding_size * 2, out_features=encoding_size),
        nn.LeakyReLU(),
        nn.Linear(in_features=encoding_size, out_features=action_size),
    ).to(device=device)

    result = IntrinsicCuriosityModule(
        feature_net, forward_head, inverse_head, discrete_action_space=False, **kwargs
    )
    return result


def build_rnd(
    env: Env[NDArray[Any], Any],
    encoding_size: int,
    lr: float = 1e-3,
    device: Device = "cpu",
    **kwargs,
):
    assert isinstance(env.observation_space, gym.spaces.Box)
    obs_size = env.observation_space.shape[0]
    target_net = nn.Sequential(
        nn.Linear(in_features=obs_size, out_features=encoding_size),
        nn.LeakyReLU(),
        nn.Linear(in_features=encoding_size, out_features=encoding_size),
        nn.LeakyReLU(),
    ).to(device=device)
    predictor_net = nn.Sequential(
        nn.Linear(in_features=obs_size, out_features=encoding_size),
        nn.LeakyReLU(),
        nn.Linear(in_features=encoding_size, out_features=encoding_size),
        nn.LeakyReLU(),
    ).to(device=device)
    return RandomNetworkDistillation(target_net, predictor_net, lr, **kwargs)


def build_disagreement(
    env: Env[NDArray[Any], NDArray[Any]],
    encoding_size: int,
    lr: float = 1e-3,
    device: Device = "cpu",
    **kwargs,
) -> Disagreement:
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # TODO: How to integrate different featurisers
    def build_forward_head():
        return nn.Sequential(
            nn.Linear(in_features=obs_size + action_size, out_features=encoding_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=encoding_size, out_features=encoding_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=encoding_size, out_features=obs_size),
        ).to(device=device)

    return Disagreement(build_forward_head, feature_net=None, lr=lr, **kwargs)


def build_logger(cfg, path: str | None = None) -> KittenLogger:
    if path is None:
        path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    return KittenLogger(
        cfg,
        cfg.algorithm.type,
        path=path,
        engine=engine_registry[cfg.log.engine.type],
        engine_kwargs={k: v for k, v in cfg.log.engine.items() if k != "type"},
    )
