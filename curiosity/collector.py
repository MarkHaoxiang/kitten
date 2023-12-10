from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional


import gymnasium as gym
from gymnasium.wrappers.autoreset import AutoResetWrapper
import torch

from curiosity.experience import ReplayBuffer
from curiosity.policy import Policy

class DataCollector(ABC):    
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
    def collect(self, *args, **kwargs) -> List:
        """ Collect data on the environment

        Returns:
            List: List of transition Tuples
        """
        raise NotImplementedError

    @abstractmethod
    def early_start(self, n: int) -> List:
        """ Fills the replay buffer with random exploration

        Args:
            n (int): number of steps

        Returns:
            List: List of transition tuples
        """
        raise NotImplementedError

    def set_policy(self, policy: Policy):
        self.policy = policy

class GymCollector(DataCollector):
    """ Gymnasium environment data collector
    """
    def __init__(self,
                 policy: Policy,
                 env: gym.Env,
                 memory: Optional[ReplayBuffer],
                 device: torch.device ='cpu'):
        if not isinstance(env, AutoResetWrapper):
            env = AutoResetWrapper(env)
        self.obs, _ = env.reset()
        self.device = device
        super().__init__(policy, env, memory)

    def collect(self, n: int, *args, **kwargs) -> List:
        with torch.no_grad():
            result = []
            obs = self.obs
            for _ in range(n):
                # Calculate actions 
                action = self.policy(obs, *args, **kwargs)
                    # Convert actions to numpy
                if isinstance(action, torch.Tensor):
                    action = action.cpu().detach().numpy()
                action = action.clip(self.env.action_space.low, self.env.action_space.high)
                # Step
                n_obs, reward, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    self.policy.reset()
                # Store buffer
                transition_tuple = (obs, action, reward, n_obs, terminated)
                result.append(transition_tuple)
                if not self.memory is None:
                    self.memory.append(transition_tuple)
                obs = n_obs
            self.obs = obs
            return result

    def early_start(self, n: int) -> List:
        policy = self.policy
        self.set_policy(Policy(lambda _: self.env.action_space.sample(), transform_obs=False))
        result = self.collect(n)
        self.policy = policy
        return result

def build_collector(policy, env, memory, device: torch.device = 'cpu') -> DataCollector:
    return GymCollector(policy, env, memory, device=device)