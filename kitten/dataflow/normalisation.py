from typing import Any, Optional, Generic, TypeVar, TypeAlias, SupportsFloat

import torch
from kitten.dataflow.interface import Transform

_ValidStatistic: TypeAlias = torch.Tensor | SupportsFloat
T = TypeVar("T", bound=_ValidStatistic)


class RunningMeanVariance(Generic[T], Transform[T, T]):
    """https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance"""

    def __init__(self) -> None:
        super().__init__()
        self._n = 0.0
        self._mean = 0.0
        self._sum_squared_difference = 0.0
        self._var = 0.0

    def add_batch(
        self, batch_size: SupportsFloat, batch_mean: T, batch_squared_difference: T
    ) -> None:
        """Adds batch statistics to track

        Args:
            batch_size: number of items within the batch.
            batch_mean: mean value of batch.
            batch_squared_difference: sum of squared differences to the mean within the batch.
        """
        batch_size = float(batch_size)
        n = self._n + batch_size
        delta = batch_mean - self._mean  # type: ignore[operator]

        self._mean = self._mean + delta * (batch_size / n)
        self._sum_squared_difference = (
            self._sum_squared_difference
            + batch_squared_difference  # type: ignore[operator]
            + delta**2 * self._n * batch_size / n
        )
        self._n = n

    def add_tensor_batch(
        self, batch: torch.Tensor, weights: Optional[torch.Tensor] = None
    ):
        """Adds tensor to track

        Args:
            batch (torch.Tensor): Tensor of shape (batch, *data)
        """
        batch = batch.to(dtype=torch.float32)
        if weights is None:
            weights = torch.ones(batch.shape[0], device=batch.device)
        weights = weights.view(
            weights.shape[0], *[1 for _ in range(len(batch.shape) - 1)]
        )

        mean = (weights * batch).mean(0)
        bsd = torch.sum((batch - mean) ** 2 * weights, 0)
        n = weights.sum().item()
        self.add_batch(n, mean, bsd)  # type: ignore[arg-type]

    def add(self, data: T) -> None:
        """Adds a single data point

        Args:
            data
        """
        self.add_batch(1, data, 0)  # type: ignore[arg-type]

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._sum_squared_difference / (self._n - 1)

    @property
    def std(self):
        return self.var**0.5

    def __str__(self) -> str:
        return f"Mean {str(self.mean)} | Std {str(self.std)}"

    def transform(
        self, data: Any, normalise_mean: bool = True, normalise_std: bool = True
    ) -> T:
        """Normalisation by statistics recorded in this instance"""
        if normalise_mean:
            data = data - self.mean
        if normalise_std:
            data = data / self.std
        return data
