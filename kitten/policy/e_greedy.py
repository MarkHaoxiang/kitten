from typing import Any, Callable

from gymnasium.spaces import Space
import numpy as np
from torch import Tensor
from numpy.typing import NDArray

from kitten.policy import Policy
from kitten.common.typing import Device
from .interface import PolicyFn


class EpsilonGreedyPolicy(Policy):
    def __init__(
        self,
        fn: PolicyFn,
        action_space: Space[Any],
        rng: np.random.Generator | None = None,
        epsilon: float = 0.1,
        device: Device = "cpu",
    ):
        """Epsilon Greedy Exploration

        Randomly sample from action_space randomly.

        Args:
            fn (Callable[..., Any]): Policy.
            action_space (Space): Action space to sample from.
        """
        super().__init__(fn, device=device)
        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng
        self._action_space = action_space
        self._epsilon = epsilon

    def __call__(self, obs: Tensor | NDArray[Any], epsilon: float | None = None):
        x = super().__call__(obs)
        if epsilon is None:
            epsilon = self._epsilon
        if self.train:
            if self._rng.random() <= epsilon:
                return self._action_space.sample()
        return x
