from abc import ABC, abstractmethod

import numpy as np
from torch import Tensor

from kitten.experience import AuxiliaryMemoryData, Transitions
from kitten.nn import Critic
from kitten.logging import Loggable


# This file contains a set of interfaces commonly used in RL
class Algorithm(Loggable, ABC):
    """Interface for RL Policy Improvement Algorithms"""

    @abstractmethod
    def update(self, batch: Transitions, aux: AuxiliaryMemoryData, step: int) -> None:
        """Policy improvement update

        Args:
            batch (Transition): Batch of transition tuples from experience
            weights (Tensor): Suggested importance sampling weights of the batch
            step (int): Training Step
        """
        raise NotImplementedError

    def policy_fn(self, s: Tensor | np.ndarray) -> Tensor:
        """A function that takes an observation and returns an action

        Args:
            s (Union[Tensor, np.ndarray]): Observation.
            batch_size (Tuple[int, ...]): Batch size of the observations.

        Returns:
            Tensor: Action.
        """
        raise NotImplementedError


class HasCritic(ABC):

    @property
    @abstractmethod
    def critic(self) -> Critic:
        raise NotImplementedError
