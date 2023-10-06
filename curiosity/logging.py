from collections.abc import Callable

import torch
from gymnasium import Env

def evaluate(env: Env, policy: Callable, repeat: int = 1):
    """ Evaluates an episode

    Args:
        env (Env): A Gym environment 
        policy (Callable): Function which accepts the observation from gym, returning an action within the action space
        repeat (int, optional): Number of attempts. Defaults to 1.

    Returns:
        _type_: _description_
    """
    total_reward = 0
    for _ in range(repeat):
        with torch.no_grad():
            env.close()
            obs, _ = env.reset()
            terminated = False
            truncated = False
            while not terminated and not truncated:
                obs, reward, terminated, truncated, _ = env.step(policy(obs))
                total_reward += reward   
    # Calculate reward 
    total_reward = total_reward / repeat
    return total_reward
