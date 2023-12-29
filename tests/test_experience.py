from typing import Sequence
import pytest

import numpy as np
import torch

from curiosity.experience.memory import ReplayBuffer, PrioritizedReplayBuffer

DEVICE = 'cpu'
torch.manual_seed(0)

class TestReplayBuffer:
    @pytest.mark.parametrize("capacity", [10,100,1000])
    @pytest.mark.parametrize("number", [10,100,1000])
    @pytest.mark.parametrize("shape", [(1,),(2,3),(4,5,(6,7))])
    def test_replay_insert(self,
                           capacity: int,
                           number: int,
                           shape: Sequence[int]):
        replay_buffer = ReplayBuffer(
            capacity=capacity,
            shape=shape,
            device=DEVICE
        )
        # Add elements one by one
        for _ in range(number):
            replay_buffer.append(tuple(torch.rand(s) for s in shape))
        # Add elements in a group
        for _ in range(5):
            data = []
            for s in shape:
                if isinstance(s, int):
                    s = (s,)
                data.append(torch.rand(number // 5, *s))
            replay_buffer.append(tuple(data))
        assert len(replay_buffer) == min(capacity, number*2), f"Incorrect number of elements contained. Expected {min(capacity, number*2)} but got {len(replay_buffer)}"

    @pytest.mark.parametrize("number", [100,1000])
    def test_replay_sample(self,
                           number: int,
                           capacity: int = 10000):
        replay_buffer = ReplayBuffer(
            capacity=capacity,
            shape=(1,),
            device=DEVICE
        )
        replay_buffer.append(
            (torch.arange(capacity).unsqueeze(-1),)
        )
        # Order should be correct
        sample = replay_buffer.sample(number, continuous = True)[0]
        for i in range(len(sample[0])-1):
            assert sample[0][i+1] == (sample[0][i]+1) % capacity, "Samples not in order"

        # Random sample
        sample = replay_buffer.sample(number, continuous=False)[0]
        for x in sample:
            assert x[0] in replay_buffer.storage[0], "Samples not in replay buffer"

class TestPrioritizedReplayBuffer:

    @pytest.mark.parametrize("capacity", [10,100,1000])
    @pytest.mark.parametrize("number", [10,100,1000])
    @pytest.mark.parametrize("shape", [(1,),(2,3),(4,5,(6,7))])
    def test_replay_insert(self,
                           capacity: int,
                           number: int,
                           shape: Sequence[int]):
        replay_buffer = PrioritizedReplayBuffer(
            error_fn=lambda x: x[0].mean().item() * 3,
            capacity=capacity,
            shape=shape,
            device=DEVICE
        )
        # Add elements one by one
        for _ in range(number):
            replay_buffer.append(tuple(torch.rand(s) for s in shape))
        # Add elements in a group
        for _ in range(5):
            data = []
            for s in shape:
                if isinstance(s, int):
                    s = (s,)
                data.append(torch.rand(number // 5, *s))
            replay_buffer.append(tuple(data))
        assert len(replay_buffer) == min(capacity, number*2), f"Incorrect number of elements contained. Expected {min(capacity, number*2)} but got {len(replay_buffer)}"
    
    @pytest.mark.parametrize("number", [100,1000])
    def test_replay_sample(self,
                           number: int,
                           capacity: int = 1000):
        replay_buffer = PrioritizedReplayBuffer(
            error_fn=lambda x: x[0].squeeze().numpy(),
            capacity=capacity,
            shape=(1,),
            device=DEVICE
        )
        replay_buffer.append(
            (torch.arange(capacity).unsqueeze(-1),)
        )
        # Random sample
        sample = replay_buffer.sample(number, continuous=False)[0]
        for x in sample:
            assert x[0] in replay_buffer.storage[0], "Samples not in replay buffer"

    def test_manual_case(self):
        shape = (1,)
        capacity = 6

        replay_buffer = PrioritizedReplayBuffer(
            error_fn=lambda x: x[0].squeeze().numpy() + 1,
            capacity=capacity,
            shape=shape,
            alpha=1,
            epsilon=0,
            device=DEVICE
        )

        replay_buffer.append(tuple(torch.ones(s, device=DEVICE) * 1 for s in shape))
        replay_buffer.append((torch.tensor([[2],[3]], device=DEVICE), ))
        assert torch.all(torch.eq(replay_buffer.storage[0].squeeze(), torch.tensor([1,2,3,0,0,0], device=DEVICE)))
        assert np.all(replay_buffer.sum_tree == np.array([3,3,0,2,1,0,0,1,1,1,0,0,0]))
        replay_buffer.sample(3)
        assert np.all(replay_buffer.sum_tree == np.array([9,9,0,5,4,0,0,2,3,4,0,0,0]))

        assert replay_buffer._get(np.array([2.0]))[0] == 0
        assert replay_buffer._get(np.array([6.0]))[0] == 2

