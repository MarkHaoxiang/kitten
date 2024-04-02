from typing import Any

import gymnasium as gym
import numpy as np


class StochasticActionWrapper(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Adds unobservable random noise to incoming actions as a source of aleatoric uncertainty"""

    def __init__(self, env: gym.Env, noise: float = 0.1):
        """A wrapper to add random noise (aleatoric uncertainty) to actions

        Args:
            env (gym.Env): The environment to apply the wrapper
            noise (float, optional): Amount of noise (Scaled Gaussian with action_scale for Box spaces). Defaults to 0.1.
        """
        if 0 > noise:
            raise ValueError(f"Noise {noise} must be above 0")

        gym.utils.RecordConstructorArgs.__init__(self, noise=noise)
        gym.ActionWrapper.__init__(self, env)
        self._noise = noise

    def action(self, action: Any) -> Any:
        """Adds random noise to the action and clips within valid bounds"""
        if isinstance(self.action_space, gym.spaces.Box):
            noise = self.np_random.normal(loc=0, scale=self._noise, size=action.shape)
            noise = noise * (self.action_space.high - self.action_space.low)
            action = action + noise
            return np.clip(action, self.action_space.low, self.action_space.high)
        else:
            # TODO: Support discrete space as well
            raise NotImplementedError()


# The wrapper below is taken from
# ===== https://github.com/MarkHaoxiang/florl/blob/main/experiments/experiment_utils.py =====
class FixedResetWrapper(gym.Wrapper):
    """Fix the reset to a single state"""

    def __init__(self, env: gym.Env):
        """Fix the reset to a single state"""
        super().__init__(env)
        self._seed = np.random.rand()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any | dict[str, Any]]:
        if seed is not None:
            self._seed = seed
        return super().reset(seed=self._seed, options=options)
