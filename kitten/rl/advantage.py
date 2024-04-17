from abc import ABC, abstractmethod

import torch
from torch import Tensor

from kitten.experience import Transitions
from kitten.nn import Value
from .common.target import generalised_advantage_estimation


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
        return generalised_advantage_estimation(
            batch, self._lmbda, self._gamma, self._value
        )
