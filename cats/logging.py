from typing import Any

import gymnasium as gym
from numpy.typing import NDArray
from torch import Tensor
from kitten.policy import Policy

from .reset import ResetActionWrapper

class ResetEvaluationPolicy(Policy):
    """ Disables reset action on 

    Args:
        Policy (_type_): _description_
    """
    
    def __init__(self,
                 env: ResetActionWrapper,
                 policy: Policy,
                 maximum_steps: int = 1000
        ) -> None:
        super().__init__(policy._fn, device=policy.device)
        self._policy = policy
        self._step_counter = 0
        self._maximum_steps = maximum_steps
        assert isinstance(env, ResetActionWrapper), "Env should be reset_action"
        self._env = env

    def fn(self, obs: Tensor):
        return self._policy.fn(obs)
    
    def reset(self) -> None:
        super().reset()
        self._step_counter = 0

    def enable_evaluation(self) -> None:
        super().enable_evaluation()
        self._policy.enable_evaluation()
    
    def disable_evaluation(self) -> None:
        super().disable_evaluation()
        self._policy.disable_evaluation()
    
    def __call__(self, obs: Tensor | NDArray[Any], *args, **kwargs):
        action = self._policy(obs, *args, **kwargs)
        self._step_counter += 1
        if self.train:
            return action
        else:
            assert isinstance(self._env.action_space, gym.spaces.Box), "Not yet implemented for non-box action spaces"
            if self._step_counter > self._maximum_steps:
                # Truncate
                action[-1] = 1
            else:
                action[-1] = 0
            return action
