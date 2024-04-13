from abc import ABC, abstractmethod

import torch
from torch import Tensor

from kitten.experience import AuxiliaryData, Transitions
from kitten.nn import Value


class AdvantageEstimator(ABC):
    @abstractmethod
    def A(self, batch: Transitions) -> Tensor:
        raise NotImplementedError


class GeneralisedAdvantageEstimator(AdvantageEstimator):
    def __init__(
        self,
        value: Value,
        exponential_weighting_factor: float = 0.9,
        discount_factor: float = 0.99,
    ) -> None:
        super().__init__()
        self._value = value
        self._lmbda = exponential_weighting_factor
        self._gamma = discount_factor

    def A(self, batch: Transitions) -> Tensor:
        with torch.no_grad():
            n = len(batch.r)
            delta = (
                batch.r
                - self._value.v(batch.s_0)
                + self._gamma * self._value.v(batch.s_1) * (~batch.d)
            )
            f = self._gamma * self._lmbda
            advantages = torch.zeros_like(batch.r, device=batch.r.get_device())
            advantages[-1] = delta[-1]
            for i in range(1, n):
                advantages[-(i + 1)] = f * advantages[-i] + delta[-(i + 1)]
            return advantages
