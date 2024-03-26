from typing import Callable

import torch
import numpy as np
from gymnasium import Env
from gymnasium.spaces.discrete import Discrete

from .memory import ReplayBuffer, PrioritizedReplayBuffer
from .collector import DataCollector, GymCollector

from kitten.dataflow.normalisation import RunningMeanVariance
from kitten.experience import Transitions
from kitten.common.typing import Device


def build_transition_from_list(updates: list, device: Device = "cpu") -> Transitions:
    """Utility to wrap collector results into a transition"""
    return build_transition_from_update(
        obs=np.array([x[0] for x in updates]),
        action=np.array([x[1] for x in updates]),
        reward=np.array([x[2] for x in updates]),
        n_obs=np.array([x[3] for x in updates]),
        terminated=np.array([x[4] for x in updates]),
        add_batch=False,
        device=device,
    )


def build_transition_from_update(
    obs,
    action,
    reward,
    n_obs,
    terminated,
    add_batch: bool = True,
    device: Device = "cpu",
) -> Transitions:
    """Utility to wrap a single gym update into a transition"""
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
    return Transitions(obs, action, reward, n_obs, terminated)


def build_replay_buffer(
    env: Env,
    capacity: int = 10000,
    type="experience_replay",
    normalise_observation: bool = False,
    error_fn: Callable | None = None,
    device: Device = "cpu",
    **kwargs,
) -> ReplayBuffer:
    """Creates a replay buffer for env

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
        _normalise_observation = [rmv, None, None, rmv, None]
    else:
        _normalise_observation = [None, None, None, None, None]

    observation_space = env.observation_space
    action_space = env.action_space
    if observation_space.shape is None or action_space.shape is None:
        raise ValueError("Invalid space")

    if type == "experience_replay":
        return ReplayBuffer(
            capacity=capacity,
            shape=(
                observation_space.shape,
                action_space.shape,
                (),
                observation_space.shape,
                (),
            ),
            dtype=(
                torch.float32,
                torch.int if discrete else torch.float32,
                torch.float32,
                torch.float32,
                torch.bool,
            ),
            transforms=_normalise_observation,
            device=device,
            **kwargs,
        )
    elif type == "prioritized_experience_replay":
        if error_fn is None:
            raise ValueError("Provide an error function for priority calculation")
        return PrioritizedReplayBuffer(
            error_fn=error_fn,
            capacity=capacity,
            shape=(
                observation_space.shape,
                action_space.shape,
                (),
                observation_space.shape,
                (),
            ),
            dtype=(
                torch.float32,
                torch.int if discrete else torch.float32,
                torch.float32,
                torch.float32,
                torch.bool,
            ),
            transforms=_normalise_observation,
            device=device,
            **kwargs,
        )
    raise NotImplementedError("Replay buffer type not supported")


def build_collector(
    policy,
    env: Env,
    memory: ReplayBuffer | None = None,
    device: Device = "cpu",
) -> DataCollector:
    """Creates a collector for env

    Args:
        policy (Policy): Collection policy.
        env (Env): Collection environment.
        memory (ReplayBuffer): Replay buffer.
        device (torch.device, optional): Memory device. Defaults to 'cpu'.

    Returns:
        DataCollector: Data collector.
    """
    return GymCollector(policy, env, memory, device=device)
