from abc import ABC, abstractmethod

import numpy as np
import torch
from kitten.common.rng import Generator

from cats.rl import QTOptCats


class TeleportStrategy(ABC):
    def __init__(self, algorithm: QTOptCats) -> None:
        super().__init__()
        self._algorithm = algorithm

    @abstractmethod
    def select(self, s: torch.Tensor) -> int:
        raise NotImplementedError


class EpsilonGreedyTeleport(TeleportStrategy):
    def __init__(
        self, algorithm: QTOptCats, rng: Generator | None = None, e: float = 0.1
    ) -> None:
        super().__init__(algorithm=algorithm)
        self._e = e
        if rng is None:
            self._rng = Generator(
                np.random.Generator(np.random.get_bit_generator()), torch.Generator()
            )
        else:
            self._rng = rng

    def select(self, s: torch.Tensor) -> int:
        v = self._algorithm.value.v(s)
        teleport_index = torch.argmax(v).item()
        if self._rng.numpy.random() < self._e:
            teleport_index = 0
        return teleport_index


class ThompsonTeleport(TeleportStrategy):
    def __init__(
        self, algorithm: QTOptCats, rng: Generator | None = None, alpha: float = 0.1
    ) -> None:
        super().__init__(algorithm=algorithm)
        self._a = alpha
        if rng is None:
            self._rng = Generator(
                np.random.Generator(np.random.get_bit_generator()), torch.Generator()
            )
        else:
            self._rng = rng

    def select(self, s: torch.Tensor) -> int:
        v = self._algorithm.value.v(s)
        with torch.no_grad():
            p = (v**self._a).cpu().numpy()
        return self._rng.numpy.choice(len(v), p=p)


class UCBTeleport(TeleportStrategy):
    def __init__(self, algorithm: QTOptCats, c=1) -> None:
        super().__init__(algorithm)
        self._c = c

    def select(
        self,
        s: torch.Tensor,
    ) -> int:
        mu, var = self._algorithm.mu_var(s)
        return np.argmax(mu + var * self._c)
