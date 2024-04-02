from abc import ABC, abstractmethod
from typing import Any, Callable, TypeAlias, Generic

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from kitten.common.typing import ObsType, ActType, Device

PolicyFn: TypeAlias = Callable[[Tensor], Tensor]


# TODO: Policy needs to be rewritten with generics to support wider range of environments
class _Policy(Generic[ObsType, ActType]):
    def __init__(self) -> None:
        super().__init__()
        self._evaluate = False

    @abstractmethod
    def __call__(self, obs: ObsType) -> ActType:
        """Actions from observations

        Args:
            inputs (Any): observations.

        Returns:
            Any: actions.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Called on episode reset"""
        pass

    def enable_evaluation(self) -> None:
        """Disable exploration for evaluation"""
        self._evaluate = True

    def disable_evaluation(self) -> None:
        """Enable exploration for training"""
        self._evaluate = False

    @property
    def evaluate(self) -> bool:
        """Returns true if policy is in evaluation mode

        Returns:
            bool: True if policy is in evaluation mode.
        """
        return self._evaluate

    @property
    def train(self) -> bool:
        """Returns true if policy is in training mode

        Returns:
            bool: True if policy is in training mode.
        """
        return not self._evaluate


class Policy:
    def __init__(
        self,
        fn: PolicyFn,
        device: Device = "cpu",
    ):
        """Policy API to pass into data collection pipeline

        Args:
            fn (Callable): Policy taking observations and returning actions.
            transform_obs (bool): Transform observations into tensors.
            normalise_obs (Optional[RunningMeanVariance]): Normalise incoming observations
        """
        self._fn = fn
        self._evaluate = False
        self.device = device

    def fn(self, obs: Tensor):
        return self._fn(obs)

    def __call__(self, obs: Tensor | NDArray[Any], *args, **kwargs):
        """Actions from observations

        Args:
            inputs (Any): observations.

        Returns:
            Any: actions.
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        return self.fn(obs)

    def reset(self) -> None:
        """Called on episode reset"""
        pass

    def enable_evaluation(self) -> None:
        """Disable exploration for evaluation"""
        self._evaluate = True

    def disable_evaluation(self) -> None:
        """Enable exploration for training"""
        self._evaluate = False

    @property
    def evaluate(self) -> bool:
        """Returns true if policy is in evaluation mode

        Returns:
            bool: True if policy is in evaluation mode.
        """
        return self._evaluate

    @property
    def train(self) -> bool:
        """Returns true if policy is in training mode

        Returns:
            bool: True if policy is in training mode.
        """
        return not self._evaluate
