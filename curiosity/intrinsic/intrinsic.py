from abc import ABC, abstractmethod

import torch
from torch import Tensor

from curiosity.experience import Transition
from curiosity.logging import Loggable

class IntrinsicReward(Loggable, ABC):
    """ Interface for intrinsic exploration rewards
    """
    def __init__(self) -> None:
        super().__init__()
        self.info = {}

    def reward(self, batch: Transition):
        """ Calculates the intrinsic exploration reward

        Args:
            batch (Transition): _description_

        Raises:
            NotImplementedError: _description_
        """
        r_i = self._reward(batch)
        self.info['r_i'] = r_i
        return r_i

    @abstractmethod
    def _reward(self, batch: Transition):
        raise NotImplementedError

    def update(self, batch: Transition, weights: Tensor, step: int):
        """ Intrinsic reward module update

        Args:
            batch (Transition): Batch of transition tuples from experience
            weights (Tensor): Suggested importance sampling weights of the batch
            step (int): Training Step
        """
        raise NotImplementedError

    def get_log(self):
        return self.info

class NoIntrinsicReward(IntrinsicReward):
    def _reward(self, batch: Transition):
        return torch.zeros(batch.r.shape[0], device=batch.r.device)
    def update(self, batch: Transition, weights: Tensor, step: int):
        pass
