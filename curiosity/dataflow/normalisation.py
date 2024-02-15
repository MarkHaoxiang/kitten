from typing import Any, Optional

import torch

from curiosity.dataflow.interface import Transform

class RunningMeanVariance(Transform):
    """ https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    def __init__(self) -> None:
        super().__init__()
        self._n = 0
        self._mean = 0
        self._sum_squared_difference = 0
        self._var = 0
    
    def add_batch(self, batch_size, batch_mean, batch_squared_difference):
        """ Adds batch statistics to track

        Args:
            batch_size: number of items within the batch.
            batch_mean: mean value of batch.
            batch_squared_difference: sum of squared differences to the mean within the batch.
        """
        n = self._n + batch_size
        delta = batch_mean - self._mean

        self._mean = self._mean + delta * (batch_size / n)
        self._sum_squared_difference = self._sum_squared_difference + batch_squared_difference + delta**2 * self._n * batch_size / n
        self._n = n

    def add_tensor_batch(self, batch: torch.Tensor, weights: Optional[torch.Tensor] = None):
        """ Adds tensor to track

        Args:
            batch (torch.Tensor): Tensor of shape (batch, *data)
        """
        if weights is None:
            weights = torch.ones(batch.shape[0], device=batch.device)
        weights = weights.view(weights.shape[0], *[1 for _ in range(len(batch.shape)-1)])

        mean = (weights * batch).mean(0)
        bsd = torch.sum((batch - mean)**2 * weights, 0)
        n = weights.sum()
        self.add_batch(n, mean, bsd)

    def add(self, data):
        """ Adds a single data point

        Args:
            data
        """
        self.add_batch(1, data, 0)

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._sum_squared_difference / (self._n-1)

    @property
    def std(self):
        return self.var ** 0.5
    
    def transform(self,
                  data: Any,
                  normalise_mean: bool = True,
                  normalise_std: bool = True) -> Any:
        """ Normalisation by statistics recorded in this instance
        """
        if normalise_mean:
            data = data - self.mean
        if normalise_std:
            data = data / self.std
        return data
