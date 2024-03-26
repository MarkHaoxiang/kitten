import os
from os.path import join
import shutil
from datetime import datetime
import time
import logging

import torch
import torch.nn as nn

logger = logging.Logger(name="KittenLogger")
log = logger.log

# KittenLogger
import importlib.util
has_wandb = importlib.util.find_spec("wandb") is not None
has_omegaconf = importlib.util.find_spec("omegaconf") is not None
if has_wandb and has_omegaconf:
    import wandb
    from omegaconf import OmegaConf, DictConfig

from kitten.common.typing import ModuleNamePairs
from .interface import Loggable


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
        if not has_wandb or not has_omegaconf:
            raise ImportError("Wandb / Omegaconf not available")

        self.cfg = cfg
        self.name = self.generate_project_name(cfg.env.name, algorithm)
        self.checkpoint_enable = cfg.log.checkpoint.enable
        self.wandb_enable = cfg.log.wandb
        self._start_time = time.time()
        self._epoch = 0
        self.models: ModuleNamePairs = []
        self.providers: list[tuple[Loggable, str]] = []

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
            config=OmegaConf.to_container(  # type: ignore[arg-type]
                cfg, resolve=True, throw_on_missing=True
            ),
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
        watch_frequency: int | None = None,
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

    def register_models(self, batch: list[tuple[nn.Module, str]], **kwargs):
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

    def checkpoint_registered(self, frame: int | None = None) -> None:
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

    def checkpoint(self, model: nn.Module, name: str, frame: int | None = None) -> None:
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


