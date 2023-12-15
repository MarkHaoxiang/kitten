from abc import ABC, abstractmethod

from torch import Tensor

from curiosity.experience import Transition
from curiosity.logging import Loggable

class Algorithm(Loggable, ABC):
    """ Interface for RL Policy Improvement Algorithms
    """
    @abstractmethod
    def update(self, batch: Transition, weights:  Tensor, step: int):
        """ Policy improvement update

        Args:
            batch (Transition): Batch of transition tuples from experience
            weights (Tensor): Suggested importance sampling weights of the batch
            step (int): Training Step
        """
        raise NotImplementedError