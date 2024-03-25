from typing import Any, Sequence, Callable
from numbers import Number

import numpy as np
import torch
from torch import Tensor

from kitten.logging import Loggable
from kitten.dataflow import Transform
from kitten.experience import AuxiliaryMemoryData
from kitten.common.typing import Device, Shape


class ReplayBuffer(Loggable):
    """Store history of episode transitions"""

    def __init__(
        self,
        capacity: int,
        shape: Sequence[Shape],
        dtype: torch.dtype | Sequence[torch.dtype] = torch.float32,
        transforms: Transform | Sequence[Transform | None] | None = None,
        device: Device = "cpu",
        **kwargs,
    ) -> None:
        """Replay buffer to store past observations

        Args:
            capacity (int): Maximum number of elements.
            shape (Sequence[int]): Shape of each storage item.
            dtype (Optional[Union[torch.dtype, Sequence[torch.dtype]]], optional): datatype of each observation item. Defaults to None.
            normalise (Union[bool, Sequence[bool]], optional): Toggle to conduct append/retrieval normalisation of store items. Defaults to False.
            device (str, optional): Storage device. Defaults to "cpu".
        """
        shape = list(shape)
        for i, s in enumerate(shape):
            if isinstance(s, int):
                shape[i] = (s,)

        self.capacity = capacity
        self.shape = shape
        self.N = len(shape)
        self.device = device
        # Dtype
        if not isinstance(dtype, Sequence):
            dtype = [dtype for _ in range(len(shape))]
        self.storage = [
            torch.zeros((capacity, *s), device=self.device, dtype=dtype[i])
            for i, s in enumerate(shape)
        ]

        # Transforms
        if isinstance(transforms, Transform):
            transforms = [transforms for _ in shape]
        self.transforms = transforms

        self._dtype = dtype
        self._random = torch.zeros(capacity, device=self.device)
        self._append_index = 0  # Position of next tensor to be inserted
        self._full = False  # Bookmark to check if storage has looped around

    def append(self, inputs: tuple[Any, ...], **kwargs) -> int:
        """Add a transition experience to the buffer

        Args:
            inputs (Tuple[Tensor): A tuple of size (types of experiences) containing (batch_size, experience data)

        Returns:
            (int): Number of inserted elements.
        """
        inputs = self._append_preprocess(inputs)
        n_insert_unclipped = len(inputs[0])
        n_insert = min(n_insert_unclipped, self.capacity)
        # Update Random IDs
        random_ids = torch.rand(n_insert, device=self.device)
        if n_insert + self._append_index <= self.capacity:
            self._random[self._append_index : self._append_index + n_insert] = (
                random_ids
            )
        else:
            self._random[self._append_index : self.capacity + 1] = random_ids[
                : self.capacity - self._append_index
            ]
            self._random[: n_insert + self._append_index - self.capacity] = random_ids[
                self.capacity - self._append_index :
            ]
        # Update Data
        for i in range(self.N):
            x = inputs[i]
            if len(x) != n_insert_unclipped:
                raise ValueError(
                    f"Input is jagged, expected batch of {n_insert_unclipped} but got {len(x)}"
                )
            # Flatten batch
            x = torch.flatten(x, end_dim=-len(x.shape))
            # Insert into storage
            x = x[-n_insert:]
            if n_insert + self._append_index <= self.capacity:
                self.storage[i][self._append_index : self._append_index + n_insert] = x
            else:
                self.storage[i][self._append_index : self.capacity + 1] = x[
                    : self.capacity - self._append_index
                ]
                self.storage[i][: n_insert + self._append_index - self.capacity] = x[
                    self.capacity - self._append_index :
                ]

        # Update internal state
        self._append_index += n_insert
        if self._append_index >= self.capacity:
            self._full = True
        self._append_index = self._append_index % self.capacity
        return n_insert

    def _append_preprocess(self, inputs: tuple[Any, ...]):
        """Utility to transform new transitions into expected format before entry"""
        res = []
        for i in range(self.N):
            x = inputs[i]
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device=self.device)
            if x.shape[-len(self.shape[i]) :] != self.shape[i]:
                raise ValueError(
                    f"Expected data ending with {self.shape[i]} but got {x.shape[-len(self.shape):]}"
                )
            if len(x.shape) == len(self.shape[i]):
                x = x.unsqueeze(0)
            res.append(x)
        return res

    def sample(self, n: int) -> tuple[tuple[Tensor, ...], AuxiliaryMemoryData]:
        """Sample from the replay buffer

        Args:
            n (int): number of samples
        Returns:
            Tuple[Tensor]: sampled elements
            AuxiliaryMemoryData
        """
        if n > len(self):
            raise ValueError(
                f"Trying to sample {n} values but only {len(self)} available"
            )
        indices = torch.randint(0, len(self), (n,), device=self.device)
        return (
            self._fetch_storage(indices),
            AuxiliaryMemoryData(
                weights=torch.ones(n, device=self.device),
                random=self._random[indices].to(self.device),
                indices=indices,
            ),
        )

    def get_log(self):
        """Returns logging info.

        Returns:
            Dict: log.
        """
        return {"size": len(self)}

    def __len__(self):
        return self.capacity if self._full else self._append_index

    def _fetch_storage(self, indices, transforms: bool = True):
        if not (isinstance(indices, torch.Tensor) or isinstance(indices, int)):
            indices = torch.tensor(indices, device=self.device)
        results = [self.storage[i][indices] for i in range(self.N)]
        # Transforms
        if transforms and not self.transforms is None:
            for i, result in enumerate(results):
                transform = self.transforms[i]
                if transform is None:
                    continue
                results[i] = transform(result)
        return tuple(results)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Implements a replay buffer with proportional based priorized sampling

    Uses an internal sum tree structure
    Schaul et al. Prioritized Experience Replay.
    """

    def __init__(
        self,
        error_fn: Callable,
        capacity: int,
        shape: Sequence[Shape],
        epsilon: float = 0.1,
        alpha: float = 0.6,
        beta_0: float = 0.4,
        dtype: torch.dtype | Sequence[torch.dtype] = torch.float32,
        transforms: Transform | Sequence[Transform | None] | None = None,
        beta_annealing_steps: int = 10000,
        device: Device = "cpu",
    ) -> None:
        super().__init__(
            capacity=capacity,
            shape=shape,
            dtype=dtype,
            transforms=transforms,
            device=device,
        )

        self.epsilon = epsilon
        self.alpha = alpha
        self.beta_0 = beta_0
        self.beta_annealing_steps = beta_annealing_steps
        self.timestep = 0
        self.sum_tree_offset = (1 << (capacity - 1).bit_length()) - 1
        self.sum_tree = np.zeros(self.sum_tree_offset + capacity)
        self.error_fn = error_fn
        self._maximum_priority = 1.0

        # Logging
        self.mean_batch_error = 0

    def append(self, inputs: tuple[Any, ...], update: bool = True, **kwargs) -> int:
        """Add transitions to the replay buffer

        Args:
            inputs (tuple[Tensor]): Transitions
            update (bool, optional): Update annealing step. Defaults to True.
        """
        initial_append_index = self._append_index
        n_insert = super().append(inputs)
        if update:
            self.timestep = min(self.beta_annealing_steps, self.timestep + n_insert)
        for i in range(initial_append_index, initial_append_index + n_insert):
            j = i % self.capacity
            self._sift_up(self._maximum_priority, j)
        return n_insert

    def sample(self, n: int) -> tuple[tuple[Tensor, ...], AuxiliaryMemoryData]:
        # Invariant checks
        if n > len(self):
            raise ValueError(
                f"Trying to sample {n} values but only {len(self)} available"
            )

        # Select random priorities from evenly spaced buckets
        priority_sum = self.sum_tree[0]
        bucket_size = priority_sum / n
        selection = (
            np.linspace(0, priority_sum - bucket_size, n)
            # TODO: Dangerous random call. Change to rng object
            + np.random.rand(n) * bucket_size
        )

        # Transform priorities to indices
        selection = self._get(selection)
        selection = torch.tensor(selection, device=self.device)

        # Calculate Weightings
        # Annealing
        beta = self.beta_0 + (self.timestep / self.beta_annealing_steps) * (
            1 - self.beta_0
        )
        # Probabities = Priority / Total Priority
        sampled_probabilities = (
            self.sum_tree[selection + self.sum_tree_offset]
        ) / self.sum_tree[0]
        # Importance sampling
        w = (len(self) * sampled_probabilities) ** (-beta)
        # Numerical stability
        w = w / np.max(w)
        w = torch.from_numpy(w).to(self.device)

        results = self._fetch_storage(selection)
        # Update errors
        errors = self.error_fn(results)
        self.mean_batch_error = np.mean(errors)
        priorities = self._calculate_priority(errors)
        for i, j in enumerate(selection):
            self._maximum_priority = max(self._maximum_priority, priorities[i])
            self._sift_up(priorities[i], j)

        # Transform indices to result tuples
        return (results, AuxiliaryMemoryData(w, self._random[selection], selection))

    def _calculate_priority(self, error):
        return (error + self.epsilon) ** self.alpha

    def _sift_up(self, value: float, i: int) -> None:
        """Updates priority of storage[i] to value

        Args:
            value (float): priority
            i (int): index within self.storage
        """
        value, i = float(value), int(i)
        index = i + self.sum_tree_offset
        change = value - self.sum_tree[index]
        self.sum_tree[index] += change
        while index > 0:
            index = (index - 1) // 2
            self.sum_tree[index] += change

    def _get(self, value: np.ndarray) -> np.ndarray:
        if np.any(value > self.sum_tree[0]):
            raise ValueError("Value out of priority range")
        result = np.zeros(value.shape[0], dtype=int)
        for _ in range((self.capacity - 1).bit_length()):
            left = self.sum_tree[result * 2 + 1]
            choice = value <= left
            result = np.where(choice, result * 2 + 1, result * 2 + 2)
            value -= (~choice) * left
        return result - self.sum_tree_offset

    def get_log(self):
        log = super().get_log()
        log["total_priority"] = self.sum_tree[0]
        log["maximum_priority"] = self._maximum_priority
        log["mean_batch_error"] = self.mean_batch_error
        log["beta"] = self.beta_0 + (self.timestep / self.beta_annealing_steps) * (
            1 - self.beta_0
        )
        return log
