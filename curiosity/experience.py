from typing import Optional, Sequence, Union, Callable
import random

from gymnasium import Env
from gymnasium.wrappers.autoreset import AutoResetWrapper
from gymnasium.spaces.discrete import Discrete
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

    def append(self, inputs: tuple[Tensor]) -> None:
        """ Add a transition experience to the buffer

        Args:
            inputs (Tensor): A tensor containing the experience. 
        """
        n_insert_unclipped = None
        n_insert = None
        for i in range(self.N):
            x = inputs[i]
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device=self.device)
            if x.shape[-len(self.shape[i]):] != self.shape[i]:
                raise ValueError(f"Expected data ending with {self.shape[i]} but got {x.shape[-len(self.shape):]}")
            if len(x.shape) == len(self.shape[i]):
                x = x.unsqueeze(0)
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

    def sample(self,
               n: int,
               continuous: bool = True) -> tuple[torch.Tensor]:
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
            return tuple(self.storage[i][start:start+n] for i in range(self.N))
        else:
            indices = random.sample(range(len(self)), n)
            return tuple(self.storage[i][torch.tensor(indices, device=self.device)] for i in range(self.N))

    def __len__(self):
        return self.capacity if self._full else self._append_index
    
def build_replay_buffer(env: Env, capacity: int = 10000, device: str = "cpu") -> ReplayBuffer:
    """ Creates a replay buffer for env

    Args:
        env (Env): Training environment
        capacity (int, optional): Capacity of the replay buffer. Defaults to 10000.
        device (str, optional): Tensor device. Defaults to "cpu".

    Returns:
        Replay Buffer: A replay buffer designed to hold tuples (State, Action, Reward, Next State, Done)
    """
    discrete = isinstance(env.action_space, Discrete)
    return ReplayBuffer(
        capacity=capacity,
        shape=(env.observation_space.shape, env.action_space.shape, (), env.observation_space.shape, ()),
        dtype=(torch.float32, torch.int if discrete else torch.float32, torch.float32, torch.float32, torch.bool),
        device=device
    )

def early_start(env: Env, memory: ReplayBuffer, steps: int, policy: Callable = None) -> ReplayBuffer:
    """ Fill the replay buffer for "steps" amount. In-place.

    Args:
        env (Env): Training Environment
        replay_buffer (ReplayBuffer): Replay buffer
        steps (int): Number of steps to run
    """
    if not isinstance(env, AutoResetWrapper):
        env = AutoResetWrapper(env)
    obs, _ = env.reset()
    for _ in range(steps):
        if not policy is None:
            action = policy(obs)
        else:
            action = env.action_space.sample()
        n_obs, reward, terminated, truncated, _ = env.step(action)
        transition_tuple = (obs, action, reward, n_obs, terminated)
        obs = n_obs
        memory.append(transition_tuple)
    return memory
