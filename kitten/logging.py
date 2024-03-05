from __future__ import annotations

from abc import ABC
import copy
from datetime import datetime
import math
import os
from os.path import join
import shutil
from typing import Callable, Dict, Optional, Tuple, List, TYPE_CHECKING
import time

import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf, DictConfig
from gymnasium import Env
from gymnasium.wrappers.record_video import RecordVideo
import wandb

# TODO: Make wandb and hydra optional dependencies

from kitten.policy import Policy

if TYPE_CHECKING:
    from kitten.rl import HasCritic


class Loggable(ABC):
    """Interface signalling a module which can be logged"""

    def get_log(self) -> Dict:
        """Collect logs for publishing

        Returns:
            Dict: Dictionary of key-value logging pairs
        """
        return {}

    def get_models(self) -> List[Tuple[nn.Module, str]]:
        """List of publishable networks with names

        Returns:
            List[Tuple[nn.Module, str]]: list.
        """
        return []


class CriticValue(Loggable):
    """The estimated critic value of the reset distribution

    #TODO: Expand to accept general value functions
    """

    def __init__(self, target: HasCritic, evaluator: KittenEvaluator) -> None:
        super().__init__()
        self._target = target
        self._evaluator = evaluator

    def get_log(self) -> Dict:
        return {
            "critic_value": self._target.critic.q(
                self._evaluator.saved_reset_states,
                self._target.policy_fn(self._evaluator.saved_reset_states),
            )
            .mean()
            .item()
        }


class KittenLogger:
    """Logging tool for reinforcement learning experiments"""

    def __init__(self, cfg: DictConfig, algorithm: str, path: str = "log") -> None:
        """Logging tool for reinforcement learning experiments.

        Wandb centric.

        Args:
            cfg (DictConfig): Experiment Configuration
            algorithm (str): Base RL algorithm
            path (str, optional): Path for saved logs. Defaults to "log".
        """
        self.cfg = cfg
        self.name = self.generate_project_name(cfg.env.name, algorithm)
        self.checkpoint_enable = cfg.log.checkpoint.enable
        self.wandb_enable = cfg.log.wandb
        self._start_time = time.time()
        self._epoch = 0
        self.models = []
        self.providers: List[Loggable] = []

        # Log
        self.path = join(path, self.name)
        cfg.log.evaluation.video.path = self.video_path
        os.makedirs(self.path)
        if self.checkpoint_enable:
            os.makedirs(join(self.path, "checkpoint"))
        OmegaConf.save(cfg, join(self.path, "config.yaml"))
        wandb.init(
            project="curiosity",
            name=self.name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=self.path,
            mode="online" if self.wandb_enable else "offline",
        )

    def generate_project_name(self, environment: str, algorithm: str) -> str:
        """Generates an identifier for the experiment

        Args:
            environment (str): Environment name
            algorithm (str): Algorithm name

        Returns:
            str: Project name
        """
        if not self.cfg.log.name:
            return "{}_{}_{}".format(
                algorithm,
                environment,
                str(datetime.now()).replace(":", "-").replace(".", "-"),
            )
        return f"{algorithm}_{self.cfg.log.name}"

    def register_model(
        self,
        model: nn.Module,
        name: str,
        watch: bool = True,
        watch_frequency: Optional[int] = None,
    ):
        """Register a torch module to checkpoint

        Args:
            model (nn.Module): model to checkpoint
            name (str): model name
        """
        if not self.checkpoint_enable:
            return

        if name[-3:] != ".pt":
            name = name + ".pt"
        self.models.append((model, name))
        self.checkpoint(model, name)
        if watch and self.wandb_enable:
            wandb.watch(
                model,
                log="all",
                log_freq=(
                    watch_frequency
                    if not watch_frequency is None
                    else self.cfg.log.frames_per_epoch
                ),
                idx=len(self.models),
            )

    def register_models(self, batch: List[Tuple[nn.Module, str]], **kwargs):
        """Register multiple models at once

        Args:
            batch (_type_):
        """
        for model, name in batch:
            self.register_model(model, name, **kwargs)

    def register_provider(self, provider, name: str):
        """Register a log provider

        Args:
            provider (_type_): should provide get_log()
            name (str): provider label
        """
        self.providers.append(((provider, name)))

    def register_providers(self, batch):
        """Register multiple providers at once

        Args:
            batch (_type_):
        """
        for provider, name in batch:
            self.register_provider(provider, name)

    def checkpoint_registered(self, frame: Optional[int] = None) -> None:
        """Utility to checkpoint registered models

        Args:
            frame (Optional[int], optional): training frame. Defaults to None.
        """
        for model, name in self.models:
            self.checkpoint(model, name, frame)

    def epoch(self) -> int:
        """Logging epoch

        Returns:
            The epoch count after logging
        """
        self._epoch += 1
        log = {"train/wall_time": self.get_wall_time()}
        for provider, name in self.providers:
            for k, v in provider.get_log().items():
                log[f"{name}/{k}"] = v
        self.log(log)
        return self._epoch

    def get_wall_time(self) -> float:
        """Gets elapsed time since creation

        Returns:
            float:
        """
        return time.time() - self._start_time

    def checkpoint(
        self, model: nn.Module, name: str, frame: Optional[int] = None
    ) -> None:
        """Utility to save a model

        Args:
            model (nn.Module) model
            name (str): identifier
            frame (Optional[int], optional): training frame.
        """
        if not self.checkpoint_enable:
            return
        path = join(self.path, "checkpoint")
        if name[-3:] != ".pt":
            name = name + ".pt"
        if frame is None:
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model, join(path, name))
        else:
            if not os.path.exists(join(path, str(frame))):
                os.makedirs(join(path, str(frame)))
            torch.save(model.state_dict(), join(path, str(frame), name))

    @property
    def video_path(self) -> str:
        return join(self.path, "video")

    def log(self, kwargs):
        """Logging a metric

        Args:
            kwargs (_type_): Passed to Wandb
        """
        wandb.log(kwargs)

    def close(self):
        """Release resources"""
        if self.wandb_enable:
            wandb.finish()

    def clear(self):
        """Deletes associated files"""
        shutil.rmtree(self.path)


class KittenEvaluator:
    """Evaluate Training Runs"""

    def __init__(
        self,
        env: Env,
        policy: Optional[Policy] = None,
        video=False,
        saved_reset_states: int = 10,
        evaluation_repeats: int = 10,
        device: str = "cpu",
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
        self.saved_reset_states = [
            self.env.reset(seed=i)[0] for i in range(saved_reset_states)
        ]
        self.saved_reset_states = torch.tensor(
            np.array(self.saved_reset_states), device=device, dtype=torch.float32
        )
        self.env = copy.deepcopy(env)

        self.repeats = evaluation_repeats
        self.policy = policy

        # Link evaluation environment with train environment statistics
        if env.spec is not None:
            for wrapper in env.spec.additional_wrappers:
                if wrapper.name == "NormalizeObservation":
                    self.env.obs_rms = env.obs_rms
                elif wrapper.name == "NormalizeReward":
                    self.env.return_rms = env.return_rms

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

        self.info = {}

    def evaluate(
        self, policy: Optional[Policy] = None, repeats: Optional[int] = None
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
            self.env, policy=policy, repeat=repeats
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


def evaluate(env: Env, policy: Callable, repeat: int = 1):
    """Evaluates an episode

    Args:
        env (Env): A Gym environment
        policy (Callable): Function which accepts the observation from gym, returning an action within the action space
        repeat (int, optional): Number of attempts. Defaults to 1.
    """
    total_reward = 0
    maximum_reward = -math.inf
    episode_length = 0
    for seed in range(repeat):
        episode_reward = 0
        with torch.no_grad():
            env.close()
            obs, _ = env.reset(seed=seed)
            terminated = False
            truncated = False
            while not terminated and not truncated:
                episode_length += 1
                action = policy(obs)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                # if action.shape == ():
                #     action = np.expand_dims(action, 0)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                episode_reward += reward
        maximum_reward = max(episode_reward, maximum_reward)
    # Calculate reward
    total_reward = total_reward / repeat
    episode_length = episode_length / repeat
    return total_reward, maximum_reward, episode_length
