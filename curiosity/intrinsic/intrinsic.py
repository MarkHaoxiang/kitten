from abc import ABC, abstractmethod

import torch
from torch import Tensor

from curiosity.experience import Transition
from curiosity.logging import Loggable

class IntrinsicReward(Loggable, ABC):
    """ Interface for intrinsic exploration rewards
    """

    @abstractmethod
    def reward(self, batch: Transition):
        """ Calculates the intrinsic exploration reward

        Args:
            batch (Transition): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def update(self, batch: Transition, weights: Tensor, step: int):
        """ Intrinsic reward module update

        Args:
            batch (Transition): Batch of transition tuples from experience
            weights (Tensor): Suggested importance sampling weights of the batch
            step (int): Training Step
        """
        raise NotImplementedError

class NoIntrinsicReward(IntrinsicReward):
    def reward(self, batch: Transition):
        return torch.zeros(batch.r.shape[0], device=batch.r.device)
    def update(self, batch: Transition, weights: Tensor, step: int):
        pass
