from typing import Callable, TypeVar, Generic

import numpy as np
import torch
import torch.nn as nn

from kitten.common.rng import Generator

T = TypeVar("T", bound=nn.Module)


class Ensemble(nn.Module, Generic[T]):

    def __init__(
        self, build_network: Callable[[], T], n: int = 5, rng: Generator | None = None
    ) -> None:
        super().__init__()

        if rng is None:
            self._rng = Generator()
        else:
            self._rng = rng

        self._n = n
        self._networks = nn.ModuleList([build_network() for _ in range(self._n)])

    def forward(self, *args, **kwargs):
        return torch.stack([net(*args, **kwargs) for net in self.networks])

    def mu_var(self, *args, **kwargs):
        res = np.array(self.forward(*args, **kwargs))
        mu, var = res.mean(axis=0), res.var(axis=0)
        return mu, var

    @property
    def networks(self) -> nn.ModuleList:
        return self._networks

    @property
    def ensemble_numer(self) -> int:
        return self._n

    def __getitem__(self, key) -> T:
        return self._networks[key]

    def sample_network(self) -> T:
        return self._rng.numpy.choice(self._networks)

    def sample_networks(self, n: int) ->  list[T]:
        """ Sample without replacement """
        return self._rng.numpy.choice(self._networks, size=2, replace=False).tolist()