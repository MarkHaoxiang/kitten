from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
from torch import Tensor

from curiosity.experience import AuxiliaryMemoryData, Transition
from curiosity.logging import Loggable

class Algorithm(Loggable, ABC):
    """ Interface for RL Policy Improvement Algorithms
    """
    @abstractmethod
    def update(self, batch: Transition, aux: AuxiliaryMemoryData, step: int) -> None:
        """ Policy improvement update

        Args:
            batch (Transition): Batch of transition tuples from experience
            weights (Tensor): Suggested importance sampling weights of the batch
            step (int): Training Step
        """
        raise NotImplementedError

    def policy_fn(self, s: Union[Tensor, np.ndarray]) -> Tensor:
        """ A function that takes an observation and returns an action

        Args:
            s (Union[Tensor, np.ndarray]): Observation.
            batch_size (Tuple[int, ...]): Batch size of the observations.

        Returns:
            Tensor: Action. 
        """
        raise NotImplementedError