from abc import ABC, abstractmethod

import torch
from torch import Tensor

from curiosity.experience import AuxiliaryMemoryData, Transition
from curiosity.logging import Loggable

class IntrinsicReward(Loggable, ABC):
    """ Interface for intrinsic exploration rewards
    """
    def __init__(self, int_coef: float = 1, ext_coef: float = 1, **kwargs) -> None:
        super().__init__()
        self.info = {}
        self._int_coef = int_coef
        self._ext_coef = ext_coef

    def reward(self, batch: Transition):
        """ Calculates the intrinsic exploration reward

        Args:
            batch (Transition): Batch of transition tuples from experience
        """
        r_i = self._reward(batch)
        self.info["mean_intrinsic_reward"] = torch.mean(r_i).item()
        self.info["max_intrinsic_reward"] = torch.max(r_i).item()
        self.info["mean_extrinsic_reward"] = torch.mean(batch.r)
        return (
            batch.r * self._ext_coef + r_i * self._int_coef, # Total Reward
            batch.r * self._ext_coef,                        # Extrinsic Reward (Scaled)
            r_i * self._int_coef                             # Intrinsic Reward
        )

    @abstractmethod
    def _reward(self, batch: Transition):
        raise NotImplementedError

    def update(self, batch: Transition, aux: AuxiliaryMemoryData, step: int):
        """ Intrinsic reward module update

        Args:
            batch (Transition): Batch of transition tuples from experience
            weights (Tensor): Suggested importance sampling weights of the batch
            step (int): Training Step
        """
        raise NotImplementedError

    def initialise(self, batch: Transition):
        """ Initialise internal state (eg. for normalisation)

        Args:
            batch: Transition
        """
        pass

    def get_log(self):
        return self.info

class NoIntrinsicReward(IntrinsicReward):
    def _reward(self, batch: Transition):
        return torch.zeros(batch.r.shape[0], device=batch.r.device)
    def update(self, batch: Transition, weights: Tensor, step: int):
        pass
