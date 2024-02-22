from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch

from curiosity.experience.memory import ReplayBuffer
from curiosity.logging import Loggable
from curiosity.policy import Policy

class DataCollector(Loggable, ABC):    
    def __init__(self, policy: Policy, env: gym.Env, memory: Optional[ReplayBuffer]):
        """ Constructs a data collector

        Args:
            policy (Callable): Policy for collection
            env (Env): Collection environment
            memory (Optional[ReplayBuffer]): If provided, add transition tuples into the replay buffer.
        """
        self.policy = policy
        self.env = env
        self.memory = memory

    @abstractmethod
    def collect(self, n: int, append_memory: bool = True, *args, **kwargs) -> List:
        """ Collect data on the environment

        Args:
            n (int): Number of steps
            append_memory (Optional[bool]): Automatically added collected transitions to the replay buffer.

        Returns:
            List: List of transition Tuples
        """
        raise NotImplementedError

    @abstractmethod
    def early_start(self, n: int) -> List:
        """ Runs the environment for a certain number of steps using a random policy.

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

class GymCollector(DataCollector):
    """ Gymnasium environment data collector
    """
    def __init__(self,
                 policy: Policy,
                 env: gym.Env,
                 memory: Optional[ReplayBuffer] = None,
                 device: torch.device ='cpu'):
        self.obs, _ = env.reset()
        self.device = device
        # Logging
        self.frame = 0
        super().__init__(policy, env, memory)

    def _policy(self, obs: np.ndarray, *args, **kwargs) -> np.ndarray:
        """ Obtain action from observations

        Args:
            obs (np.ndarray): Gymnasium environment

        Returns:
            np.ndarray: _description_
        """
        action = self.policy(obs, *args, **kwargs)
        if isinstance(action, torch.Tensor):
            action = action.cpu().detach().numpy()
        if isinstance(self.env.action_space, gym.spaces.Box):
            action = action.clip(self.env.action_space.low, self.env.action_space.high)
        return action

    def collect(self,
                n: int,
                append_memory: bool = True,
                early_start: bool = False,
                *args, **kwargs) -> List:
        if not early_start:
            self.frame += n
        with torch.no_grad():
            result = []
            obs = self.obs
            for _ in range(n):
                # Calculate actions 
                action = self._policy(obs, *args, **kwargs)
                # Step
                n_obs, reward, terminated, truncated, _ = self.env.step(action)
                transition_tuple = (obs, action, reward, n_obs, terminated)
                # Store buffer
                if not self.memory is None and append_memory:
                    self.memory.append(transition_tuple, update=not early_start)
                # Add Truncation info back in
                transition_tuple = (obs, action, reward, n_obs, terminated, truncated)
                result.append(transition_tuple)
                # Reset
                if terminated or truncated:
                    n_obs, _ = self.env.reset()
                    self.policy.reset()
                # Crucial Step
                obs = n_obs

            self.obs = obs
            return result

    def early_start(self, n: int, dry_run: bool = False) -> List:
        """ Runs the environment for a certain number of steps using a random policy.

        For example, to fill the replay buffer or initialise normalisations.

        Args:
            n (int): number of steps.
            dry_run (bool): run extra steps to first initialise any Gymnasium wrappers.

        Returns:
            List: List of transition tuples
        """
        # Set random policy
        policy = self.policy
        self.set_policy(Policy(lambda _: self.env.action_space.sample(), transform_obs=False))
        # Run
        if dry_run:
            self.collect(n, append_memory=False, early_start=True)
        result = self.collect(n, append_memory=True, early_start=True)
        # Restore initial policy
        self.policy = policy
        return result

    def get_log(self):
        return {
            "frame": self.frame
        }
