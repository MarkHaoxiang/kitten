from typing import Any, Callable

from numpy.typing import NDArray
import gymnasium as gym
from gymnasium.spaces import Space
import pink  # type: ignore[import-untyped]
import numpy as np
import torch
from torch import Tensor

from kitten.policy import Policy
from kitten.common import Generator
from .interface import PolicyFn


class ColoredNoisePolicy(Policy):
    def __init__(
        self,
        fn: PolicyFn,
        action_space: Space[Any],
        episode_length: int | None = None,
        scale: float | NDArray[Any] | torch.Tensor = 1.0,
        beta: float = 1,
        rng: Generator | np.random.Generator | None = None,
        device: str = "cpu",
    ):
        """Pink Noise Exploration

        Eberhard et al.
        Pink Noise Is All You Need: Colored Noise Exploration in Deep Reinforcement Learning

        Args:
            fn (Callable[..., Any]): Policy.
            action_space (Space): Environment action space.
            episode_length (int): Maximum episode length.
            scale (float): Exploration factor scaling
            beta (float, optional): Correlation parameter. 0 is white noise, 2 is OU. Defaults to 1.
            device (str, optional): Training device. Defaults to "cpu".
        """
        super().__init__(fn, device=device)
        # Calculate shape
        if isinstance(action_space, gym.spaces.Box):
            size = action_space.shape
        elif isinstance(action_space, gym.spaces.Discrete):
            raise NotImplementedError("Action space not yet supported")
        else:
            raise ValueError("Action space not supported")
        if episode_length is None:
            episode_length = 1000
        if isinstance(rng, Generator):
            rng = rng.numpy
        self._noise = pink.ColoredNoiseProcess(
            beta=beta, size=(*size, episode_length), scale=1, rng=rng
        )
        self._device = device
        self.scale = torch.tensor(scale, device=self._device)

    def __call__(self, obs: Tensor | NDArray[Any]) -> Any:
        action = super().__call__(obs)
        if self.train:
            noise = self._noise.sample(1)
            if isinstance(action, torch.Tensor):
                noise = torch.tensor(noise, device=self._device).reshape(action.shape)
                noise = noise * self.scale
            action += noise
        return action

    def reset(self):
        self._noise.reset()
