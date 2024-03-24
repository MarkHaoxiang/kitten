from typing import Callable

from gymnasium.spaces import Space
import numpy as np
from torch import Tensor

from kitten.policy import Policy


class EpsilonGreedyPolicy(Policy):
    def __init__(
        self,
        fn: Callable[[Tensor | np.ndarray], Tensor],
        action_space: Space,
        rng: np.random.Generator | None = None,
        device: str = "cpu",
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

    def __call__(self, obs: Tensor | np.ndarray, epsilon: float = 0.1):
        x = super().__call__(obs)
        if self.train:
            if self._rng.random() <= epsilon:
                return self._action_space.sample()
        return x
