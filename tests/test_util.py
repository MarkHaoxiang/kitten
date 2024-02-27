import random
import pytest
import unittest

import torch
import gymnasium as gym

from kitten.util import global_seed
from kitten.dataflow.normalisation import RunningMeanVariance

class TestDeterminism:
    @pytest.mark.parametrize("seed", [14,60])
    @pytest.mark.parametrize("environment", ["Pendulum-v1", "MountainCarContinuous-v0"])
    def test_global_seed(self, seed: int, environment: str):
        env = gym.make(environment)

        generated_1 = {}
        global_seed(seed, env)
        generated_1['random'] = random.random()
        generated_1['torch'] = torch.rand(1)
        generated_1["env_state"] = env.reset()[0]
        generated_1["env_space"] = env.action_space.sample()

        generated_2 = {}
        global_seed(seed, env)
        generated_2['random'] = random.random()
        generated_2['torch'] = torch.rand(1)
        generated_2["env_state"] = env.reset()[0]
        generated_2["env_space"] = env.action_space.sample()

        for (k, v) in generated_1.items():
            try:
                assert v == generated_2[k]
            except:
                assert (v == generated_2[k]).all()

class TestRunningMeanStd(unittest.TestCase):
    def test_tensor(self):
        torch.manual_seed(0)
        stat = RunningMeanVariance()
        data = torch.rand((7, 3, 2))

        for x in data:
            stat.add_tensor_batch(x)

        data = data.reshape((data.shape[0] * data.shape[1], data.shape[-1]))
        mean = data.mean(0)
        assert torch.allclose(mean, stat.mean)
        var = ((data - mean.unsqueeze(0))**2).sum(0) / (data.shape[0]-1)
        assert torch.allclose(var, stat.var)

    def test_number(self):
        stat = RunningMeanVariance()
        data = [random.random() for _ in range(10)]

        # True    
        mean = sum(data) / len(data)
        var = 0
        for x in data:
            var += (x-mean)**2 / (len(data) - 1)

        # Calculated
        for x in data:
            stat.add(x)

        self.assertAlmostEqual(stat.mean, mean)
        self.assertAlmostEqual(stat.var, var)
