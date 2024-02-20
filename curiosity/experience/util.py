from typing import Any, Optional, Callable, List

import torch
from gymnasium import Env
from gymnasium.spaces.discrete import Discrete

from .memory import ReplayBuffer, PrioritizedReplayBuffer
from .collector import DataCollector, GymCollector

from curiosity.dataflow.normalisation import RunningMeanVariance
from curiosity.experience import Transition

def build_transition_from_list(updates: List, device: str = "cpu") -> Transition:
    """ Utility to wrap collector results into a transition
    """
    return build_transition_from_update(
        obs=[x[0] for x in updates],
        action=[x[1] for x in updates],
        reward=[x[2] for x in updates],
        n_obs=[x[3] for x in updates],
        terminated=[x[4] for x in updates],
        add_batch=False,
        device=device
    )


def build_transition_from_update(obs,
                                 action,
                                 reward,
                                 n_obs,
                                 terminated,
                                 add_batch: bool = True,
                                 device: str = "cpu") -> Transition:
    """ Utility to wrap a single gym update into a transition
    """
    obs = torch.tensor(obs, device=device, dtype=torch.float32)
    action = torch.tensor(action, device=device, dtype=torch.float32)
    reward = torch.tensor(reward, device=device, dtype=torch.float32)
    n_obs = torch.tensor(n_obs, device=device, dtype=torch.float32)
    terminated = torch.tensor(terminated, device=device)
    if add_batch:
        obs = obs.unsqueeze(0)
        action = action.unsqueeze(0)
        n_obs = n_obs.unsqueeze(0)
        terminated = terminated.unsqueeze(0)
    return Transition(obs, action, reward, n_obs, terminated)

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
        rmv = RunningMeanVariance()
        normalise_observation = [rmv, None, None, rmv, None]
    else:
        normalise_observation = [None, None, None, None, None]

    if type == "experience_replay":
        return ReplayBuffer(
            capacity=capacity,
            shape=(env.observation_space.shape, env.action_space.shape, (), env.observation_space.shape, ()),
            dtype=(torch.float32, torch.int if discrete else torch.float32, torch.float32, torch.float32, torch.bool),
            transforms=normalise_observation,
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
            transforms=normalise_observation,
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
