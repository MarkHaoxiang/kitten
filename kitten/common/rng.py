from __future__ import annotations

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

    def __getattr__(self, name: str):
        try:
            return self._np_rng.__getattribute__(name)
        except AttributeError:
            try:
                return self._torch_rng.__getattribute__(name)
            except AttributeError:
                raise AttributeError(name)

    @property
    def torch(self) -> torch.Generator:
        return self._torch_rng

    @property
    def numpy(self) -> np.random.Generator:
        return self._np_rng

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
