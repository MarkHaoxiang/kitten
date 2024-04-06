from typing import Any, Callable

import torch
from torch import Tensor
import numpy as np
from numpy.typing import NDArray
from gymnasium import Env
from gymnasium.spaces.discrete import Discrete
from jaxtyping import Float32
import hydra

from kitten.dataflow.normalisation import RunningMeanVariance
from kitten.experience import Transitions
from kitten.common.typing import ActType, Device, Shape, Log
from .memory import ReplayBuffer, PrioritizedReplayBuffer
from .collector import DataCollector, GymCollector
from .interface import AuxiliaryMemoryData, Memory


class TransitionReplayBuffer(
    Memory[tuple[Any, ...], tuple[Transitions, AuxiliaryMemoryData]]
):
    def __init__(self, rb: ReplayBuffer) -> None:
        super().__init__()
        self._rb = rb

    def append(self, data: tuple[Any, ...], **kwargs) -> int:
        return self._rb.append(data, **kwargs)

    def sample(self, n: int, **kwargs):
        batch, aux = self._rb.sample(n)
        batch = Transitions(*batch)
        return batch, aux

    def get_log(self) -> Log:
        return self._rb.get_log()

    def __len__(self):
        return len(self._rb)

    @property
    def rb(self) -> ReplayBuffer:
        return self._rb

    @staticmethod
    def shape(env: Env[Any, Any], enable_termination: bool = True) -> tuple[Shape, ...]:
        obs_space, act_space = env.observation_space, env.action_space
        if obs_space.shape is None or act_space.shape is None:
            raise ValueError("Invalid space does not have a shape")
        if enable_termination:
            return (obs_space.shape, act_space.shape, (), obs_space.shape, (), ())
        else:
            return (
                obs_space.shape,
                act_space.shape,
                (),
                obs_space.shape,
                (),
            )

    @staticmethod
    def dtype(
        env: Env[Any, Any], enable_termination: bool = True
    ) -> tuple[torch.dtype, ...]:
        discrete = isinstance(env.action_space, Discrete)
        if enable_termination:
            return (
                torch.float32,
                torch.int if discrete else torch.float32,
                torch.float32,
                torch.float32,
                torch.bool,
                torch.bool,
            )
        else:
            return (
                torch.float32,
                torch.int if discrete else torch.float32,
                torch.float32,
                torch.float32,
                torch.bool,
            )


def build_transition_from_list(
    updates: list[tuple[Any, Any, Any, Any, Any]], device: Device = "cpu"
) -> Transitions:
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
    action = torch.tensor(action, device=device)
    reward = torch.tensor(reward, device=device, dtype=torch.float32)
    n_obs = torch.tensor(n_obs, device=device, dtype=torch.float32)
    terminated = torch.tensor(terminated, device=device)
    if add_batch:
        obs = obs.unsqueeze(0)
        action = action.unsqueeze(0)
        n_obs = n_obs.unsqueeze(0)
        reward = reward.unsqueeze(0)
        terminated = terminated.unsqueeze(0)
    return Transitions(obs, action, reward, n_obs, terminated)


def build_replay_buffer(
    env: Env[NDArray[Any], ActType],
    capacity: int = 10000,
    type="experience_replay",
    normalise_observation: bool = False,
    error_fn: Callable[[tuple[Tensor, ...]], NDArray[Any]] | None = None,
    device: Device = "cpu",
    **kwargs,
) -> tuple[
    TransitionReplayBuffer, RunningMeanVariance[Float32[torch.Tensor, "..."]] | None
]:
    """Creates a replay buffer for env

    Args:
        env (Env): Training environment
        capacity (int, optional): Capacity of the replay buffer. Defaults to 10000.
        device (str, optional): Tensor device. Defaults to "cpu".

    Returns:
        Replay Buffer: A replay buffer designed to hold tuples (State, Action, Reward, Next State, Done)
    """
    if normalise_observation:
        rmv = RunningMeanVariance[Float32[torch.Tensor, "..."]]()
        _normalise_observation = [rmv, None, None, rmv, None, None]
    else:
        _normalise_observation = [None, None, None, None, None, None]

    observation_space = env.observation_space
    action_space = env.action_space
    if observation_space.shape is None or action_space.shape is None:
        raise ValueError("Invalid space")

    if type == "experience_replay":
        return (
            TransitionReplayBuffer(
                ReplayBuffer(
                    capacity=capacity,
                    shape=TransitionReplayBuffer.shape(env),
                    dtype=TransitionReplayBuffer.dtype(env),
                    transforms=_normalise_observation,
                    device=device,
                    **kwargs,
                )
            ),
            rmv,
        )
    elif type == "prioritized_experience_replay":
        if error_fn is None:
            raise ValueError("Provide an error function for priority calculation")
        return (
            TransitionReplayBuffer(
                PrioritizedReplayBuffer(
                    error_fn=error_fn,
                    capacity=capacity,
                    shape=TransitionReplayBuffer.shape(env),
                    dtype=TransitionReplayBuffer.dtype(env),
                    transforms=_normalise_observation,
                    device=device,
                    **kwargs,
                )
            ),
            rmv,
        )
    raise NotImplementedError("Replay buffer type not supported")


def build_collector(
    policy,
    env: Env[NDArray[Any], ActType],
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
