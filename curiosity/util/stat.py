import torch

class RunningMeanVariance:
    """ https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    def __init__(self) -> None:
        self._n = 0
        self._mean = 0
        self._sum_squared_difference = 0
        self._var = 0
    
    def add_batch(self, batch_size, batch_mean, batch_squared_difference):
        n = self._n + batch_size
        delta = batch_mean - self._mean

        self._mean = self._mean + delta * (batch_size / n)
        self._sum_squared_difference = self._sum_squared_difference + batch_squared_difference + delta**2 * self._n * batch_size / n
        self._n = n

    def add_tensor_batch(self, batch: torch.Tensor):
        """ Adds tensor to track

        Args:
            batch (torch.Tensor): Tensor of shape (batch, *data)
        """
        mean = batch.mean(0)
        bsd = torch.sum((batch - mean)**2, 0)
        n = batch.shape[0]
        self.add_batch(n, mean, bsd)

    def add(self, data):
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
