from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import gymnasium as gym
from gymnasium.wrappers.autoreset import AutoResetWrapper
import torch

from curiosity.experience import ReplayBuffer

class DataCollector(ABC):    
    def __init__(self, policy: Callable, env, memory: Optional[ReplayBuffer]):
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

    def set_policy(self, policy: Callable):
        self.policy = policy

class GymCollector(DataCollector):
    """ Gymnasium environment data collector
    """
    def __init__(self,
                 policy: Callable,
                 env: gym.Env,
                 memory: Optional[ReplayBuffer],
                 transform_obs: bool = True,
                 device: torch.device ='cpu'):
        if not isinstance(env, AutoResetWrapper):
            env = AutoResetWrapper(env)
        self._transform_obs = transform_obs
        self.obs, _ = env.reset()
        self.device = device
        super().__init__(policy, env, memory)

    def collect(self, n: int) -> List:
        with torch.no_grad():
            result = []
            obs = self.obs
            for _ in range(n):
                # Calculate actions
                if self._transform_obs:
                    action = self.policy(torch.tensor(obs, device=self.device, dtype=torch.float32))
                else:
                    action = self.policy(obs)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().detach().numpy().clip(self.env.action_space.low, self.env.action_space.high)
                # Step
                n_obs, reward, terminated, truncated, _ = self.env.step(action)
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
        self.set_policy(lambda _: self.env.action_space.sample())
        result = self.collect(n)
        self.policy = policy
        return result

def build_collector(policy, env, memory, device: torch.device = 'cpu') -> DataCollector:
    return GymCollector(policy, env, memory, transform_obs=True, device=device)