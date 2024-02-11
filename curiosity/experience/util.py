from typing import Any, Optional, Callable
from collections import namedtuple

import torch
from gymnasium import Env
from gymnasium.spaces.discrete import Discrete

from .memory import ReplayBuffer, PrioritizedReplayBuffer
from .collector import DataCollector, GymCollector

# A class representing a standard Markov Decision Process Transition
Transition = namedtuple('Transition', ["s_0", "a", "r", "s_1", "d"])

def build_transition_from_update(obs,
                                        action,
                                        reward,
                                        n_obs,
                                        terminated,
                                        device: str = "cpu"):
    """ Utility to wrap a single gym update into a transition
    """
    return Transition(
        torch.tensor(obs, device=device).unsqueeze(0),
        torch.tensor(action, device=device).unsqueeze(0),
        torch.tensor(reward, device=device).unsqueeze(0),
        torch.tensor(n_obs, device=device).unsqueeze(0),
        torch.tensor(terminated, device=device).unsqueeze(0)
    )

def build_replay_buffer(env: Env,
                        capacity: int = 10000,
                        type = "experience_replay",
                        normalise_observation: bool = False,
                        error_fn: Optional[Callable] = None,
                        device: str = "cpu",
                        **kwargs) -> ReplayBuffer:
    """ Creates a replay buffer for env

    Args:
        env (Env): Training environment
        capacity (int, optional): Capacity of the replay buffer. Defaults to 10000.
        device (str, optional): Tensor device. Defaults to "cpu".

    Returns:
        Replay Buffer: A replay buffer designed to hold tuples (State, Action, Reward, Next State, Done)
    """
    discrete = isinstance(env.action_space, Discrete)
    if normalise_observation:
        normalise_observation = [True, False, False, True, False]

    if type == "experience_replay":
        return ReplayBuffer(
            capacity=capacity,
            shape=(env.observation_space.shape, env.action_space.shape, (), env.observation_space.shape, ()),
            dtype=(torch.float32, torch.int if discrete else torch.float32, torch.float32, torch.float32, torch.bool),
            normalise=normalise_observation,
            device=device,
            **kwargs
        )
    elif type == "prioritized_experience_replay":
        if error_fn is None:
            raise ValueError("Provide an error function for priority calculation")
        return PrioritizedReplayBuffer(
            error_fn=error_fn,
            capacity=capacity,
            shape=(env.observation_space.shape, env.action_space.shape, (), env.observation_space.shape, ()),
            dtype=(torch.float32, torch.int if discrete else torch.float32, torch.float32, torch.float32, torch.bool),
            normalise=normalise_observation,
            device=device,
            **kwargs
        )
    raise NotImplementedError("Replay buffer type not supported")

def build_collector(policy, env: Env, memory: ReplayBuffer, device: torch.device = 'cpu') -> DataCollector:
    """ Creates a collector for env

    Args:
        policy (Policy): Collection policy.
        env (Env): Collection environment.
        memory (ReplayBuffer): Replay buffer.
        device (torch.device, optional): Memory device. Defaults to 'cpu'.

    Returns:
        DataCollector: Data collector.
    """
    return GymCollector(policy, env, memory, device=device)
