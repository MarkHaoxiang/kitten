import random
import pytest

import torch
import gymnasium as gym

from curiosity.util import global_seed

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
            print(v)
            try:
                assert v == generated_2[k]
            except:
                assert (v == generated_2[k]).all()
