from abc import ABC, abstractmethod
import copy
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
import gymnasium as gym
from kitten.common.rng import Generator
from kitten.experience.memory import ReplayBuffer
from kitten.experience.interface import Memory

from cats.rl import QTOptCats

class TeleportStrategy(ABC):
    def __init__(self, algorithm: QTOptCats) -> None:
        super().__init__()
        self._algorithm = algorithm

    @abstractmethod
    def select(self, s: torch.Tensor) -> int:
        raise NotImplementedError


class EpsilonGreedyTeleport(TeleportStrategy):
    def __init__(
        self, algorithm: QTOptCats, rng: Generator | None = None, e: float = 0.1
    ) -> None:
        super().__init__(algorithm=algorithm)
        self._e = e
        if rng is None:
            self._rng = Generator(
                np.random.Generator(np.random.get_bit_generator()), torch.Generator()
            )
        else:
            self._rng = rng

    def select(self, s: torch.Tensor) -> int:
        v = self._algorithm.value.v(s)
        teleport_index = torch.argmax(v).item()
        if self._rng.numpy.random() < self._e:
            teleport_index = 0
        return teleport_index


class ThompsonTeleport(TeleportStrategy):
    def __init__(
        self, algorithm: QTOptCats, rng: Generator | None = None, alpha: float = 0.1
    ) -> None:
        super().__init__(algorithm=algorithm)
        self._a = alpha
        if rng is None:
            self._rng = Generator(
                np.random.Generator(np.random.get_bit_generator()), torch.Generator()
            )
        else:
            self._rng = rng

    def select(self, s: torch.Tensor) -> int:
        v = self._algorithm.value.v(s)
        with torch.no_grad():
            p = (v**self._a).cpu().numpy()
        p = p.squeeze() / p.sum()
        return self._rng.numpy.choice(len(v), p=p)


class UCBTeleport(TeleportStrategy):
    def __init__(self, algorithm: QTOptCats, c=1) -> None:
        super().__init__(algorithm)
        self._c = c

    def select(
        self,
        s: torch.Tensor,
    ) -> int:
        mu, var = self._algorithm.mu_var(s)
        return np.argmax(mu + var * self._c)

class TeleportMemory(ABC):
    @abstractmethod
    def update(self, env: gym.Env, obs: NDArray[Any]):
        """ Environment step

        Args:
            env (gym.Env): Current collector env.
            obs (NDArray[Any]): Observation
        """
        raise NotImplementedError

    @abstractmethod
    def targets(self):
        """ List of potential teleportation targets
        """
        raise NotImplementedError

    @abstractmethod
    def select(self, tid: int) -> gym.Env:
        """ Teleport to target tid

        Returns:
            gym.Env: Teleportation state
        """
        raise NotImplementedError

class LatestEpisodeTeleportMemory(TeleportMemory):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.teleport_target_observations: list[NDArray[Any]] = []
        self.teleport_target_saves: list[gym.Env] = []
        self.episode_step = 0
        self.device=device

    def update(self, env: gym.Env, obs: NDArray[Any]):
        self.teleport_target_saves.append(copy.deepcopy(env))
        self.teleport_target_observations.append(obs)
        self.episode_step += 1
    
    def targets(self):
        return torch.tensor(
            self.teleport_target_observations[:self.episode_step],
            dtype=torch.float32,
            device=self.device
        )

    def select(self, tid: int) -> tuple[gym.Env, NDArray[Any]]:
        self.episode_step = tid+1
        self.teleport_target_observations = self.teleport_target_observations[:tid+1]
        self.teleport_target_saves = self.teleport_target_saves[:tid+1]
        return copy.deepcopy(self.teleport_target_saves[tid]), self.teleport_target_observations[tid]
    
class FIFOTeleportMemory(TeleportMemory):
    def __init__(self, env: gym.Env, capacity: int = 1024, device: str = "cpu") -> None:
        super().__init__()
        self.teleport_target_observations = ReplayBuffer(
            capacity=capacity, 
            shape=(env.observation_space.shape, ),
            dtype=(torch.float32, ),
            device=device
        )
