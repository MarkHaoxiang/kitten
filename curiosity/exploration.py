import random

import numpy as np
import torch
from gymnasium.spaces import Space

def epsilon_greedy(x, action_space: Space, epsilon: float):
    assert 0 <= epsilon <= 1
    if random.random() <= epsilon:
        return action_space.sample()
    else:
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()
        return x