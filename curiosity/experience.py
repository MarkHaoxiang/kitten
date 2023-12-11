from typing import Optional, Sequence, Union, Callable, Tuple
import random

from gymnasium import Env
from gymnasium.spaces.discrete import Discrete
import numpy as np
import torch
from torch import Tensor

class ReplayBuffer:
    """ Store history of episode transitions
    """
    def __init__(self,
                 capacity: int,
                 shape: Sequence[int],
                 device: str = "cpu",
                 dtype: Optional[Union[torch.dtype, Sequence[torch.dtype]]] = None) -> None:
        shape = list(shape)
        for i, s in enumerate(shape):
            if isinstance(s, int):
                shape[i] = (s,)

        self.capacity = capacity
        self.shape = shape
        self.N = len(shape)
        self.device = device

        if hasattr(dtype, "__getitem__"):
            self.storage = [torch.zeros((capacity, *s), device=self.device, dtype=dtype[i]) for i, s in enumerate(shape)]
        else:
            self.storage = [torch.zeros((capacity, *s), device=self.device, dtype=dtype) for s in shape]
        self._append_index = 0 # Position of next tensor to be inserted
        self._full = False # Bookmark to check if storage has looped around

    def append(self, inputs: tuple[Tensor]) -> int:
        """ Add a transition experience to the buffer

        Args:
            inputs (Tensor): A tensor of size (types of experiences) containing (batch_size, experience data)

        Returns:
            (int): Number of inserted elements.
        """
        inputs = self._append_preprocess(inputs)
        n_insert_unclipped = None
        n_insert = None
        for i in range(self.N):
            x = inputs[i]
            if n_insert is None:
                n_insert_unclipped = len(x)
                n_insert = min(len(x), self.capacity)
            elif len(x) != n_insert_unclipped:
                raise ValueError(f"Input is jagged, expected batch of {n_insert_unclipped} but got {len(x)}")
            # Flatten batch
            x = torch.flatten(x, end_dim=-len(x.shape))
            # Insert into storage
            x = x[-n_insert:]
            if n_insert + self._append_index <= self.capacity:
                self.storage[i][self._append_index:self._append_index+n_insert] = x
            else:
                self.storage[i][self._append_index:self.capacity+1] = x[:self.capacity-self._append_index]
                self.storage[i][:n_insert+self._append_index-self.capacity] = x[self.capacity-self._append_index:]

        self._append_index += n_insert
        if self._append_index >= self.capacity:
            self._full = True
        self._append_index = self._append_index % self.capacity
        return n_insert

    def _append_preprocess(self, inputs: tuple[Tensor]):
        """ Utility to transform new transitions into expected format before entry
        """
        res = []
        for i in range(self.N):
            x = inputs[i]
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device=self.device)
            if x.shape[-len(self.shape[i]):] != self.shape[i]:
                raise ValueError(f"Expected data ending with {self.shape[i]} but got {x.shape[-len(self.shape):]}")
            if len(x.shape) == len(self.shape[i]):
                x = x.unsqueeze(0)
            res.append(x)
        return res

    def sample(self,
               n: int,
               continuous: bool = False) -> Tuple[torch.Tensor]:
        """ Sample from the replay buffer

        Args:
            n (int): number of samples
            continuous (bool, optional): If true, samples are ordered as the order they are inserted. Defaults to True.

        Returns:
            Tuple[Tensor]: sampled elements
            Tensor: recommended importance weightings
        """
        if n > len(self):
            raise ValueError(f"Trying to sample {n} values but only {len(self)} available")
        if continuous:
            start = random.randint(0, len(self)-n)
            return tuple(self.storage[i][start:start+n] for i in range(self.N))
        else:
            indices = random.sample(range(len(self)), n)
            return self._fetch_storage(indices), torch.ones(n, device=self.device)

    def __len__(self):
        return self.capacity if self._full else self._append_index
 
    def _fetch_storage(self, indices):
        if not (isinstance(indices, torch.Tensor) or isinstance(indices, int)):
            indices = torch.tensor(indices, device=self.device)
        return tuple(self.storage[i][indices] for i in range(self.N))


class PrioritizedReplayBuffer(ReplayBuffer):
    """ Implements a replay buffer with proportional based priorized sampling

    Uses an internal sum tree structure
    Schaul et al. Prioritized Experience Replay. 
    """
    def __init__(self,
                 error_fn: Callable,
                 capacity: int,
                 shape: Sequence[int],
                 epsilon: float = 0.1,
                 alpha: float = 0.6,
                 beta: float = 0,
                 dtype: Optional[Union[torch.dtype, Sequence[torch.dtype]]] = None,
                 device: str = "cpu") -> None:
        super().__init__(capacity, shape, device, dtype)
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.sum_tree_offset = (1 << (capacity-1).bit_length()) - 1
        self.sum_tree = np.zeros(self.sum_tree_offset+capacity)
        self.error_fn = error_fn

    def append(self, inputs: tuple[Tensor]) -> None:
        initial_append_index = self._append_index
        n_insert = super().append(inputs)

        for i in range(initial_append_index, initial_append_index + n_insert):
            j = i % self.capacity
                # Insert maximum priority
            priority = self._calculate_priority(self.error_fn(self._fetch_storage([j])))
            #priority = np.max([priority[0], np.max(self.sum_tree[self.sum_tree_offset:])])
            #priority = np.max([100, np.max(self.sum_tree[self.sum_tree_offset:])])
            self._sift_up(priority, j)

    def sample(self, n: int, continuous: bool = False) -> tuple[Tensor]:
        # Invariant checks
        if n > len(self):
            return ValueError(f"Trying to sample {n} values but only {len(self)} available")
        if continuous:
            raise NotImplementedError("Continuous samples not implemented with priority")

        # Select random priorities from evenly spaced buckets
        priority_sum = self.sum_tree[0]
        bucket_size = priority_sum / n
        selection = np.linspace(0, priority_sum - bucket_size, n) + np.random.rand(n) * bucket_size

        # Transform priorities to indices
        selection = self._get(selection)

        # Calculate Weightings
        sampled_priorities = self.sum_tree[selection + self.sum_tree_offset]
        sampled_probabilities = sampled_priorities / self.sum_tree[0]
        w = 1 / (len(self) * sampled_probabilities)**self.beta
        w = w / np.max(w)
        w = torch.from_numpy(w).to(self.device)

        results = self._fetch_storage(selection)
        # Update errors 
        priorities = self._calculate_priority(self.error_fn(results))
        for i, j in enumerate(selection):
            self._sift_up(priorities[i], j)

        # Transform indices to result tuples
        return results, w

    def _calculate_priority(self, error):
        return (error + self.epsilon) ** self.alpha 

    def _sift_up(self, value: float, i: int) -> None:
        """ Updates priority of storage[i] to value

        Args:
            value (float): priority
            i (int): index within self.storage
        """
        value, i = float(value), int(i)
        index = i + self.sum_tree_offset
        change = value - self.sum_tree[index]
        self.sum_tree[index] += change
        while index > 0:
            index = (index-1) // 2
            self.sum_tree[index] += change

    def _get(self, value: np.ndarray) -> np.ndarray:
        if np.any(value > self.sum_tree[0]):
            raise ValueError("Value out of priority range")
        result = np.zeros(value.shape[0], dtype=int)
        for _ in range((self.capacity-1).bit_length()):
            left = self.sum_tree[result*2+1]
            choice = (value <= left)
            result = np.where(choice, result*2+1, result*2+2)
            value -= (~choice) * left
        return result - self.sum_tree_offset

def build_replay_buffer(env: Env,
                        capacity: int = 10000,
                        type = "experience_replay",
                        error_fn: Optional[Callable] = None,
                        device: str = "cpu") -> ReplayBuffer:
    """ Creates a replay buffer for env

    Args:
        env (Env): Training environment
        capacity (int, optional): Capacity of the replay buffer. Defaults to 10000.
        device (str, optional): Tensor device. Defaults to "cpu".

    Returns:
        Replay Buffer: A replay buffer designed to hold tuples (State, Action, Reward, Next State, Done)
    """
    discrete = isinstance(env.action_space, Discrete)
    if type == "experience_replay":
        return ReplayBuffer(
            capacity=capacity,
            shape=(env.observation_space.shape, env.action_space.shape, (), env.observation_space.shape, ()),
            dtype=(torch.float32, torch.int if discrete else torch.float32, torch.float32, torch.float32, torch.bool),
            device=device
        )
    elif type == "prioritized_experience_replay":
        if error_fn is None:
            raise ValueError("Provide an error function for priority calculation")
        return PrioritizedReplayBuffer(
            error_fn=error_fn,
            capacity=capacity,
            shape=(env.observation_space.shape, env.action_space.shape, (), env.observation_space.shape, ()),
            dtype=(torch.float32, torch.int if discrete else torch.float32, torch.float32, torch.float32, torch.bool),
            device=device
        )
    raise NotImplementedError("Replay buffer type not supported")