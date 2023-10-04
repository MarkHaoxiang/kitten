from typing import Optional
import random

import torch
from torch import Tensor

class ReplayBuffer:
    """ Store history of episode transitions
    """
    def __init__(self,
                 capacity: int,
                 shape: int,
                 device: str = "cpu",
                 dtype: Optional[torch.dtype] = None) -> None:
        self.capacity = capacity
        self.shape = shape
        self.device = device
        self.storage = torch.zeros(
            size = (self.capacity, *self.shape),
            device=self.device,
            dtype=dtype
        )
        self._append_index = 0 # Position of next tensor to be inserted
        self._full = False # Bookmark to check if storage has looped around

    def append(self, x: Tensor) -> None:
        """ Add a transition experience to the buffer

        Args:
            x (Tensor): A tensor containing the experience. 
        """
        if x.shape[-len(self.shape):] != self.shape:
            raise ValueError(f"Expected data ending with {self.shape} but got {x.shape}")
        
        if len(x.shape) == len(self.shape):
            x = x.unsqueeze(0)

        # Flatten batch
        x = torch.flatten(x, end_dim=-len(x.shape))

        # Insert into storage
        n_insert = min(len(x), self.capacity)
        x = x[-n_insert:]
        if n_insert + self._append_index <= self.capacity:
            self.storage[self._append_index:self._append_index+n_insert] = x
        else:
            self.storage[self._append_index:self.capacity+1] = x[:self.capacity-self._append_index]
            self.storage[:n_insert+self._append_index-self.capacity] = x[self.capacity-self._append_index:]
        
        self._append_index += n_insert
        if self._append_index >= self.capacity:
            self._full = True
        self._append_index = self._append_index % self.capacity

    def sample(self,
               n: int,
               continuous: bool = True) -> torch.Tensor:
        """ Sample from the replay buffer

        Args:
            n (int): number of samples
            continuous (bool, optional): If true, samples are ordered as the order they are inserted. Defaults to True.

        Returns:
            Tensor: sampled elements
        """
        if n > len(self):
            raise ValueError(f"Trying to sample {n} values but only {len(self)} available")
        if continuous:
            start = random.randint(0, len(self)-n)
            return self.storage[start:start+n]
        else:
            indices = random.sample(range(len(self)), n)
            return self.storage[torch.tensor(indices, device=self.device)]

    def __len__(self):
        return self.capacity if self._full else self._append_index

    def __getattr__(self, name: str):
        return self.storage.__getattribute__(name)
