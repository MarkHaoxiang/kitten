from __future__ import annotations

import copy
import random

import torch
import numpy as np

class Generator:
    """ Encapsulates Torch and Numpy generators to maintain consistent state across different modules
    """
    def __init__(self, 
                 np_rng: np.random.Generator,
                 torch_rng: torch.Generator) -> None:
        """  Encapsulates Torch and Numpy generators to maintain consistent state across different modules

        Args:
            np_rng (np.random.Generator): Numpy Generator.
            torch_rng (torch.Generator): Torch Generator.
        """
        self._np_rng = np_rng
        self._torch_rng = torch_rng

    def __getstate__(self):
        result = copy.copy(vars(self))
        result["_torch_rng"] = self._torch_rng.get_state()
        return result

    def __setstate__(self, state):
        objects = vars(self)
        self._torch_rng = torch.default_generator
        torch_rng_state = state.pop("_torch_rng")
        self._torch_rng.set_state(torch_rng_state)
        objects.update(state)

    def __getattr__(self, name: str):
        return self._np_rng.__getattribute__(name)

    @property
    def torch(self) -> torch.Generator:
        return self._torch_rng

    @property
    def numpy(self) -> np.random.Generator:
        return self._np_rng
    
    # def __getstate__(self):
    #     result = vars(self)
    #     result["_torch_rng"] = self._torch_rng.get_state()
    #     return result
# 
    # def __setstate__(self, state):
    #     self._torch_rng = torch.default_generator
    #     self._torch_rng.set_state(state.pop("_torch_rng"))
    #     result = vars(self)
    #     result.pop("_torch_rng")
    #     result.update(state)

def global_seed(seed: int, *envs):
    """Utility to help set determinism

    Args:
        seed (int): seed for rng generation
        envs (Env): all environments
    """
    random.seed(seed)
    torch_rng = torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    np.random.seed(seed)
    for env in envs:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return Generator(
        np_rng=np_rng,
        torch_rng=torch_rng,
    )