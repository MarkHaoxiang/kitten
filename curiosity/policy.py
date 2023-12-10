from typing import Any, Callable, Optional
import random

import gymnasium as gym
from gymnasium.spaces import Space
import pink
import torch

class Policy:
    def __init__(self, fn: Callable, transform_obs: bool = True, device="cpu"):
        """ Policy API to pass into data collection pipeline

        Args:
            fn (Callable): Policy taking observations and returning actions.
        """
        self.fn = fn
        self._evaluate = False
        self._transform_obs = transform_obs
        self.device = device

    def __call__(self, obs: Any, *args, **kwargs) -> Any:
        """ Actions from observations

        Args:
            inputs (Any): observations.

        Returns:
            Any: actions.
        """
        if self._transform_obs:
            obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        return self.fn(obs)

    def reset(self) -> None:
        """ Called on episode reset
        """
        pass

    def enable_evaluation(self) -> None:
        """Disable exploration for evaluation
        """
        self._evaluate = True

    def disable_evaluation(self) -> None:
        """Enable exploration for training
        """
        self._evaluate = False

    @property
    def evaluate(self) -> bool:
        return self._evaluate
    
    @property
    def train(self) -> bool:
        return not self._evaluate

class EpsilonGreedyPolicy(Policy):
    def __init__(self, fn: Callable[..., Any], action_space: Space, device: str = "cpu"):
        """ Epsilon Greedy Exploration

        Randomly sample from action_space randomly.

        Args:
            fn (Callable[..., Any]): Policy.
            action_space (Space): Action space to sample from.
        """
        super().__init__(fn, device=device)
        self._action_space = action_space

    def __call__(self, obs: Any, epsilon: float = 0.1):
        x = super().__call__(obs)
        if self.train:
            if random.random() <= epsilon:
                return self._action_space.sample()
        return x

class ColoredNoisePolicy(Policy):
    def __init__(self,
                 fn: Callable[..., Any],
                 action_space: Space,
                 episode_length: Optional[int],
                 exploration_factor: float = 1,
                 beta: float = 1,
                 rng = None,
                 device: str = "cpu"):
        """ Pink Noise Exploration

        Eberhard et al.
        Pink Noise Is All You Need: Colored Noise Exploration in Deep Reinforcement Learning

        Args:
            fn (Callable[..., Any]): Policy.
            action_space (Space): Environment action space.
            episode_length (int): Maximum episode length.
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
        self._noise = pink.ColoredNoiseProcess(
            beta=beta,
            size=(*size, episode_length),
            scale=exploration_factor,
            rng = rng
        )
        self._device = device

    def __call__(self, obs: Any) -> Any:
        action =  super().__call__(obs)
        if self.train:
            noise = self._noise.sample(1)
            if isinstance(action, torch.Tensor):
                noise = torch.tensor(noise, device=self._device)
            action += noise
        return action

    def reset(self):
        self._noise.reset()