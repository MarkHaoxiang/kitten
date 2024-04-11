import copy
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from kitten.policy import Policy
from kitten.experience.memory import ReplayBuffer
from kitten.common.typing import Device
from kitten.common.rng import Generator
from kitten.experience.collector import GymCollector


class ResetMemory:
    def __init__(
        self,
        env: gym.Env,
        capacity: int = 1024,
        rng: Generator = None,
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.env = copy.deepcopy(env)
        self.rng = rng
        self.reset_target_observations = ReplayBuffer(
            capacity=capacity,
            shape=(env.observation_space.shape,),
            dtype=(torch.float32,),
            device=device,
        )
        self.reset_envs = [None for _ in range(self.capacity)]
        self.refresh()

    def refresh(self) -> None:
        assert self.reset_target_observations._append_index == 0
        self.reset_envs = []
        for _ in range(self.capacity):
            obs, _ = self.env.reset()
            self.reset_target_observations.append((obs,))
            self.reset_envs.append(copy.deepcopy(self.env))

    def targets(self):
        return self.reset_target_observations.storage[0]

    def select(self, tid: int, collector: GymCollector) -> tuple[gym.Env, NDArray[Any]]:
        obs = (
            self.reset_target_observations._fetch_storage(indices=tid)[0].cpu().numpy()
        )
        env = copy.deepcopy(self.reset_envs[tid])
        collector.env, collector.obs = env, obs
        collector.env.np_random = self.rng.build_generator().numpy
        return env, obs


class ResetActionWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper adds an extra action option to the environment, and taking it send a truncation signal"""

    def __init__(self, env: gym.Env, penalty: float = 0.0, deterministic: bool = False):
        assert penalty >= 0.0, f"Penalty {penalty} must be greater than or equal to 0"
        gym.utils.RecordConstructorArgs.__init__(
            self, penalty=penalty, deterministic=deterministic
        )
        gym.Wrapper.__init__(self, env)

        # Change action spaces
        if isinstance(self.action_space, gym.spaces.Box):
            assert (
                len(self.action_space.shape) == 1
            ), f"shape of action space should be 1d"
            self.action_space = gym.spaces.Box(
                low=np.concatenate((self.action_space.low, [0])),
                high=np.concatenate((self.action_space.high, [1])),
                shape=(self.action_space.shape[0] + 1,),
                dtype=self.action_space.dtype,
            )
        elif isinstance(self.action_space, gym.spaces.Discrete):
            self.action_space = gym.spaces.Discrete(
                n=self.action_space.n + 1, start=self.action_space.start
            )
        else:
            # TODO
            raise NotImplementedError()

        # TODO: Add indicator observation for True on reset_possible, otherwise False

        self._previous_obs = None
        self._penalty = penalty
        self._deterministic = deterministic

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._previous_obs = obs
        return obs, info

    def _env_step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._previous_obs = observation
        return observation, reward, terminated, truncated, info

    def step(self, action: Any):
        if isinstance(self.action_space, gym.spaces.Box):
            if self._deterministic:
                manual_truncation = action[-1] > 0.5
            else:
                manual_truncation = self.np_random.random() > action[-1]
            obs, r, d, t, i = self._env_step(action[:-1])
            r = r - self._penalty if manual_truncation else r
            return obs, r, d, t or manual_truncation, i
        elif isinstance(self.action_space, gym.spaces.Discrete):
            if action == self.action_space.start + self.action_space.n - 1:
                return self._previous_obs, -self._penalty, False, True, {}
            else:
                return self._env_step(action)
        else:
            # TODO
            raise NotImplementedError()


class ResetPolicy(Policy):
    """Manually overrides resets for...

    Removing resets on evaluation episodes (and adds truncation back)
    Enforcing minimum exploration time after resets

    Args:
        Policy (_type_): _description_
    """

    def __init__(
        self,
        env: ResetActionWrapper,
        policy: Policy,
        minimum_exploration_time: int = 0,
        maximum_eval_steps: int = 1000,
    ) -> None:
        super().__init__(policy._fn, device=policy.device)
        self._policy = policy
        self._step_counter = 0
        self._minimum_exploration_time = minimum_exploration_time
        self._maximum_eval_steps = maximum_eval_steps
        assert isinstance(env, ResetActionWrapper), "Env should be reset_action"
        self._env = env

    @property
    def fn(self):
        return self._policy.fn

    def reset(self) -> None:
        super().reset()
        self._policy.reset()
        self._step_counter = 0

    def enable_evaluation(self) -> None:
        super().enable_evaluation()
        self._policy.enable_evaluation()

    def disable_evaluation(self) -> None:
        super().disable_evaluation()
        self._policy.disable_evaluation()

    def __call__(self, obs: Tensor | NDArray[Any], *args, **kwargs):
        assert isinstance(
            self._env.action_space, gym.spaces.Box
        ), "Not yet implemented for non-box action spaces"
        action = self._policy(obs, *args, **kwargs)
        self._step_counter += 1
        if self.train:
            if self._step_counter < self._minimum_exploration_time:
                action[-1] = 0
            return action
        else:
            if self._step_counter > self._maximum_eval_steps:
                # Truncate
                action[-1] = 1
            else:
                action[-1] = 0
            return action
