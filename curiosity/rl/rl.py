from abc import ABC, abstractmethod
from typing import Callable

from curiosity.experience import AuxiliaryMemoryData, Transition
from curiosity.logging import Loggable

class Algorithm(Loggable, ABC):
    """ Interface for RL Policy Improvement Algorithms
    """
    @abstractmethod
    def update(self, batch: Transition, aux: AuxiliaryMemoryData, step: int):
        """ Policy improvement update

        Args:
            batch (Transition): Batch of transition tuples from experience
            weights (Tensor): Suggested importance sampling weights of the batch
            step (int): Training Step
        """
        raise NotImplementedError

    @property
    def policy_fn(self) -> Callable:
        """ A function which takes in an observation and returns an action
        """
        raise NotImplementedError