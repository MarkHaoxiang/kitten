from abc import ABC, abstractmethod
from typing import Any, Generic

import gymnasium as gym
from numpy.typing import NDArray
import torch

from kitten.experience.memory import ReplayBuffer
from kitten.logging import Loggable
from kitten.policy import Policy
from kitten.common.typing import Device, ActType
from kitten.common.lib import policy_wrapper


class DataCollector(Loggable, ABC):
    def __init__(
        self,
        policy: Policy,
        env: gym.Env[NDArray[Any], ActType],
        memory: ReplayBuffer | None,
    ):
        """Constructs a data collector

        Args:
            policy (Callable): Policy for collection
            env (Env): Collection environment
            memory (Optional[ReplayBuffer]): If provided, add transition tuples into the replay buffer.
        """
        self.policy = policy
        self.env = env
        self.memory = memory

    @abstractmethod
    def collect(self, n: int, append_memory: bool = True, *args, **kwargs) -> list[Any]:
        """Collect data on the environment

        Args:
            n (int): Number of steps
            append_memory (Optional[bool]): Automatically added collected transitions to the replay buffer.

        Returns:
            List: List of transition Tuples
        """
        raise NotImplementedError

    @abstractmethod
    def early_start(self, n: int) -> list[Any]:
        """Runs the environment for a certain number of steps using a random policy.

        For example, to fill the replay buffer or initialise normalisations.

        Args:
            n (int): number of steps

        Returns:
            List: List of transition tuples
        """
        raise NotImplementedError

    def set_policy(self, policy: Policy):
        self.policy = policy

    def get_log(self):
        return {}


class GymCollector(Generic[ActType], DataCollector):
    """Gymnasium environment data collector"""

    def __init__(
        self,
        policy: Policy,
        env: gym.Env[NDArray[Any], ActType],
        memory: ReplayBuffer | None = None,
        device: Device = "cpu",
    ):
        self.obs, _ = env.reset()
        self.device = device
        # Logging
        self.frame = 0
        super().__init__(policy, env, memory)

    def collect(
        self,
        n: int,
        append_memory: bool = True,
        early_start: bool = False,
        single_episode: bool = False,
        *args,
        **kwargs,
    ) -> list[tuple[NDArray[Any], ActType, float, NDArray[Any], bool, bool]]:
        policy = policy_wrapper(self.policy, self.env)
        if not early_start:
            self.frame += n
        with torch.no_grad():
            result = []
            s = self.obs
            for _ in range(n):
                # Calculate actions
                a = policy(s, *args, **kwargs)
                # Step
                s_1, r, d, t, _ = self.env.step(a)
                transition = (
                    s,
                    a,
                    float(r),
                    s_1,
                    d,
                    t,
                )
                # Store buffer
                result.append(transition)
                if not self.memory is None and append_memory:
                    self.memory.append(transition, update=not early_start)
                # Reset
                if d or t:
                    s_1, _ = self.env.reset()
                    self.policy.reset()
                # Crucial Step
                s = s_1
                if (d or t) and single_episode:
                    self.obs = s
                    return result

            self.obs = s
            return result

    def early_start(
        self, n: int, dry_run: bool = False, append_memory: bool = True, 
    ) -> list[tuple[NDArray[Any], ActType, float, NDArray[Any], bool, bool]]:
        """Runs the environment for a certain number of steps using a random policy.

        For example, to fill the replay buffer or initialise normalisations.

        Args:
            n (int): number of steps.
            dry_run (bool): run extra steps to first initialise any Gymnasium wrappers.

        Returns:
            List: List of transition tuples
        """
        # Set random policy
        policy = self.policy
        self.set_policy(Policy(lambda _: self.env.action_space.sample()))
        # Run
        if dry_run:
            self.collect(n, append_memory=False, early_start=True)
        result = self.collect(n, append_memory=append_memory, early_start=True)
        # Restore initial policy
        self.policy = policy
        return result

    def get_log(self):
        return {"frame": self.frame}
