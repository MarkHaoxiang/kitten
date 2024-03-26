from typing import Any, Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from kitten.common.typing import Device

PolicyFn: TypeAlias = Callable[[Tensor], Tensor]


class Policy:
    def __init__(
        self,
        fn: PolicyFn,
        device: Device ="cpu",
    ):
        """Policy API to pass into data collection pipeline

        Args:
            fn (Callable): Policy taking observations and returning actions.
            transform_obs (bool): Transform observations into tensors.
            normalise_obs (Optional[RunningMeanVariance]): Normalise incoming observations
        """
        self.fn = fn
        self._evaluate = False
        self.device = device

    def __call__(self, obs: Tensor | NDArray[Any], *args, **kwargs) -> Any:
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
