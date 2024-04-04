from typing import Any, Callable

import numpy as np
import torch
from gymnasium import Env
from gymnasium.spaces import Box
from kitten.common.typing import ActType, ObsType


def policy_wrapper(
    policy: Callable[[ObsType], Any], env: Env[ObsType, ActType]
) -> Callable[[ObsType], ActType]:
    def pi(obs):
        action = policy(obs)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if action.shape == () and isinstance(env.action_space, Box):
            action = np.expand_dims(action, 0)
        return action

    return pi
