import random
from gymnasium.spaces import Space

def epsilon_greedy(x, action_space: Space, epsilon: float):
    assert 0 <= epsilon <= 1
    if random.random() <= epsilon:
        return action_space.sample()
    else:
        return x
