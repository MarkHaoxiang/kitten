from typing import Sequence
import pytest

import torch

from curiosity.experience import ReplayBuffer

DEVICE = 'cpu'
torch.manual_seed(0)

class TestReplayBuffer:
    @pytest.mark.parametrize("capacity", [10,100,1000,10000])
    @pytest.mark.parametrize("number", [10,100,1000,10000])
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
        sample = replay_buffer.sample(number, continuous = True)
        for i in range(len(sample[0])-1):
            assert sample[0][i+1] == (sample[0][i]+1) % capacity, "Samples not in order"

        # Random sample
        sample = replay_buffer.sample(number, continuous=False)
        for x in sample:
            assert x[0] in replay_buffer.storage[0], "Samples not in replay buffer"
