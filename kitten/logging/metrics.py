import copy
import math
from logging import WARNING
from typing import Any, Callable

import torch
from torch import Tensor
import numpy as np
from gymnasium import Env
from gymnasium.wrappers.record_video import RecordVideo

from kitten.policy import Policy
from kitten.common.typing import Log, ObsType, ActType, Device
from kitten.common.lib import policy_wrapper
from kitten.nn import HasValue

from .engine import log
from .interface import Loggable


class KittenEvaluator(Loggable):
    """Evaluate Training Runs"""

    def __init__(
        self,
        env: Env[Any, Any],
        policy: Policy | None = None,
        video=False,
        saved_reset_states: int = 10,
        evaluation_repeats: int = 10,
        device: Device = "cpu",
    ) -> None:
        """Evaluate Training Runs.

        Use the evaluator on every evaluation epoch.

        Args:
            env (Env): Training environment.
            policy (Policy): Evaluation policy.
            video (_type_): Log evaluation videos.
            saved_reset_states (int, optional): Set of stored observations from the reset distribution. Defaults to 10.
            device (str, optional): Device. Defaults to "cpu".
        """
        # Hack for seeding
        self.env = copy.deepcopy(env)
        self.saved_reset_states = torch.tensor(
            data=np.array(
                [self.env.reset(seed=i)[0] for i in range(saved_reset_states)]
            ),
            device=device,
            dtype=torch.float32,
        )

        self.repeats = evaluation_repeats
        self.policy = policy

        # Link evaluation environment with train environment statistics
        # Deprecated
        # Use Kitten Transforms instead
        # if env.spec is not None:
        #     for wrapper in env.spec.additional_wrappers:
        #         if wrapper.name == "NormalizeObservation":
        #             self.env.obs_rms = env.obs_rms
        #         elif wrapper.name == "NormalizeReward":
        #             self.env.return_rms = env.return_rms

        if video != False and video.enable:
            if env.render_mode != "rgb_array":
                raise ValueError(
                    f"Video mode requires rgb_array render mode but got {env.render_mode}"
                )
            self.env = RecordVideo(
                env=self.env,
                video_folder=video.path,
                episode_trigger=lambda x: x % video.frames_per_video == 0,
                disable_logger=True,
            )

        self.info: Log = {}

    def evaluate(
        self, policy: Policy | None = None, repeats: int | None = None
    ) -> float:
        """Evaluation of a policy on the environment

        Args:
            repeats (int): Number of repeats. Defaults to value set at initialisation.

        Returns:
            float: Mean reward
        """
        # Initialise policy and repeats
        if policy is None:
            policy = self.policy
        if policy is None:
            raise ValueError("Provide a policy to evaluate")
        if repeats is None:
            repeats = self.repeats

        policy.enable_evaluation()
        reward, maximum_reward, episode_length = evaluate(
            self.env, policy=policy_wrapper(policy, self.env), repeat=repeats
        )
        policy.disable_evaluation()

        self.info["reward"] = reward
        self.info["max_reward"] = maximum_reward
        self.info["episode_length"] = episode_length

        return reward

    def get_log(self):
        return self.info

    def __getattr__(self, name: str):
        return self.env.__getattribute__(name)

    def close(self):
        self.env.close()

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)


def evaluate(
    env: Env[ObsType, ActType], policy: Callable[[ObsType], ActType], repeat: int = 1
) -> tuple[float, float, float]:
    """Evaluates an episode

    Args:
        env (Env): A Gym environment
        policy (Callable): Function which accepts the observation from gym, returning an action within the action space
        repeat (int, optional): Number of attempts. Defaults to 1.
    """
    total_reward = 0.0
    episode_length = 0.0
    maximum_reward = -math.inf
    for seed in range(repeat):
        episode_reward = 0.0
        with torch.no_grad():
            env.close()
            obs, _ = env.reset(seed=seed)
            terminated = False
            truncated = False
            while not terminated and not truncated:
                episode_length += 1
                action = policy(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                episode_reward += float(reward)
        maximum_reward = max(episode_reward, maximum_reward)
    # Calculate reward
    total_reward = total_reward / repeat
    episode_length = episode_length / repeat
    return total_reward, maximum_reward, episode_length


class EstimatedValue(Loggable):
    """The estimated value of the reset distribution"""

    def __init__(self, target: Any, evaluator: KittenEvaluator) -> None:
        super().__init__()
        self._target: None | HasValue = None
        if not isinstance(target, HasValue):
            log(WARNING, f"{target} does not have a Value module")
        else:
            self._target = target
            self._evaluator = evaluator

    def estimate_value(self, obs: Tensor) -> float:
        assert isinstance(self._target, HasValue), "Doesn't have a value function"
        return self._target.value.v(obs).mean().item()

    def get_log(self) -> Log:
        if isinstance(self._target, HasValue):
            return {
                "critic_value": self.estimate_value(self._evaluator.saved_reset_states)
            }
        return {}
