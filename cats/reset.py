from typing import Any

import numpy as np
import gymnasium as gym


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

# Old Version With Check Resets

# class ResetActionWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
#     """This wrapper adds an extra action option to the environment, and taking it send a truncation signal"""
# 
#     def __init__(
#         self, env: gym.Env, check_frequency: int = 1, deterministic: bool = False
#     ):
#         gym.utils.RecordConstructorArgs.__init__(
#             self, check_frequency=check_frequency, deterministic=deterministic
#         )
#         gym.Wrapper.__init__(self, env)
# 
#         # Change action spaces
#         if isinstance(self.action_space, gym.spaces.Box):
#             assert (
#                 len(self.action_space.shape) == 1
#             ), f"shape of action space should be 1d"
#             self.action_space = gym.spaces.Box(
#                 low=np.concatenate((self.action_space.low, [0])),
#                 high=np.concatenate((self.action_space.high, [1])),
#                 shape=(self.action_space.shape[0] + 1,),
#                 dtype=self.action_space.dtype,
#             )
#         elif isinstance(self.action_space, gym.spaces.Discrete):
#             self.action_space = gym.spaces.Discrete(
#                 n=self.action_space.n + 1, start=self.action_space.start
#             )
#         else:
#             # TODO
#             raise NotImplementedError()
# 
#         # TODO: Add indicator observation for True on reset_possible, otherwise False
# 
#         self._previous_obs = None
#         self._check_frequency = check_frequency
#         self._current_step = 0
#         self._deterministic = deterministic
# 
#     def reset_current_step(self):
#         self._current_step = 0
# 
#     def reset(self, *, seed=None, options=None):
#         obs, info = self.env.reset(seed=seed, options=options)
#         self._previous_obs = obs
#         self._current_step = 0
#         return obs, info
# 
#     def _env_step(self, action: Any):
#         observation, reward, terminated, truncated, info = self.env.step(action)
#         self._previous_obs = observation
#         return observation, reward, terminated, truncated, info
# 
#     def step(self, action: Any):
#         self._current_step += 1
#         should_check = (
#             self._current_step % self._check_frequency == 0 and self._current_step > 0
#         )
#         if isinstance(self.action_space, gym.spaces.Box):
#             if should_check:
#                 if self._deterministic:
#                     manual_truncation = action[-1] > 0.5
#                 else:
#                     manual_truncation = self.np_random.random() > action[-1]
#             else:
#                 manual_truncation = False
#             observation, reward, terminated, truncated, info = self._env_step(
#                 action[:-1]
#             )
#             return observation, reward, terminated, truncated or manual_truncation, info
#         elif isinstance(self.action_space, gym.spaces.Discrete):
#             if (
#                 should_check
#                 and action == self.action_space.start + self.action_space.n - 1
#             ):
#                 return self._previous_obs, 0, False, True, {}
#             else:
#                 return self._env_step(action)
#         else:
#             # TODO
#             raise NotImplementedError()