from gymnasium import Env
import numpy as np
import torch

from kitten.intrinsic.rnd import RandomNetworkDistillation
from ..cats import CatsExperiment


# Aims to evaluate which trained intrinsic model is the best
def evaluate_rnd(experiment: CatsExperiment, samples: int = 10000) -> float:
    env = experiment.env
    device = experiment.device
    env.observation_space.seed(0)
    s = torch.tensor(
        np.array([env.observation_space.sample() for _ in range(samples)])
    ).to(device=device)
    s = experiment.rmv.transform(s)
    rnd = experiment.intrinsic
    assert isinstance(rnd, RandomNetworkDistillation)
    return rnd.forward(s).mean().item()
