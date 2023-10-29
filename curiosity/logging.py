import argparse
import copy
from datetime import datetime
import math
import warnings
import os
import shutil
from typing import Callable, Dict, Optional
import json

import torch
import torch.nn as nn
import numpy as np
from gymnasium import Env
from gymnasium.wrappers.record_video import RecordVideo
import wandb

class CuriosityArgumentParser:
    """ Argument parser for training scripts
    """
    def __init__(self, prog: str, description: str):
        self.parser = argparse.ArgumentParser(prog=prog, description=description)
        self.parser.add_argument(
            '-c',
            "--config",
            required=False,
            help="The configuration file to pass in."
        )
        self.parser.add_argument(
            "-n",
            "--name",
            required=False,
            help="Project name"
        )
        self.args = self.parser.parse_args()

    def generate_project_name(self, environment: str, algorithm: str) -> str:
        """ Generates an identifier for the experiment

        Args:
            environment (str): Environment name
            algorithm (str): Algorithm name

        Returns:
            str: Project name
        """
        if self.args.name is None:
            return "{}_{}_{}".format(algorithm, environment, str(datetime.now()).replace(":","-").replace(".","-"))
        return f"{algorithm}_{self.args.name}"

    def run(self, train: Callable):
        """ Executes a training loop with this configuration

        Args:
            train (Callable): Handle to the training function
        """
        if self.args.config is None:
            config = {}
        else:
            with open(self.args.config, 'r') as f:
                config = json.load(f)
        train(config, **config)


    def __getattr__(self, name: str):               
        return self.parser.__getattribute__(name)

class EvaluationEnv:
    """ Evaluate training runs and keep logs
    """
    def __init__(self,
                 env: Env,
                 project_name: str,
                 config: Dict,
                 path: str = "log",
                 video: Optional[int] = 10000,
                 wandb_enable: bool = False,
                 saved_reset_states: int = 10,
                 device: str ="cpu"):
        super().__init__()
        # State
        self.project_name = project_name
        self.wandb_enable = wandb_enable
        self.config = config
        self.env = copy.deepcopy(env)
        self.path = os.path.join(path, project_name)
        self.saved_reset_states = []
        for _ in range(saved_reset_states):
            obs, _ = self.env.reset()
            self.saved_reset_states.append(obs)
        self.saved_reset_states = torch.tensor(np.array(self.saved_reset_states), device=device, dtype=torch.float32)

        # Log
        os.makedirs(self.path)
        os.makedirs(os.path.join(self.path, "checkpoint"))

        if video:
            if env.render_mode != "rgb_array":
                raise ValueError(f"Video mode requires rgb_array render mode but got {env.render_mode}")
            self.env = RecordVideo(
                self.env,
                os.path.join(self.path, "video"),
                episode_trigger=lambda x: x%video == 0,
                disable_logger=True
            )
        
        with open(os.path.join(self.path, "config.json"), "w") as f:
            json.dump(config, f)

        wandb.init(
            project = "curiosity",
            name = self.project_name,
            config = self.config,
            dir = self.path,
            mode = "online" if wandb_enable else "offline"
        )

        # Checkpoints
        self.models = []

    def register(self, model: nn.Module, name: str) -> None:
        """Register a torch module to checkpoint

        Args:
            model (nn.Module): model to checkpoint
            name (str): model name
        """
        if name[-3:] != ".pt":
            name = name + ".pt"
        self.models.append((model, name))

    def checkpoint_registered(self, frame: Optional[int] = None) -> None:
        """ Utility to checkpoint registered models

        Args:
            frame (Optional[int], optional): training frame. Defaults to None.
        """
        for (model, name) in self.models:
            self.checkpoint(model, name, frame)

    def checkpoint(self, model: nn.Module, name: str, frame: Optional[int] = None) -> None:
        """Utility to save a model

        Args:
            model (nn.Module) model
            name (str): identifier
            frame (Optional[int], optional): training frame.
        """
        path = os.path.join(self.path, "checkpoint")
        if name[-3:] != ".pt":
            name = name + ".pt"
        if frame is None:
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model, os.path.join(path, name))
        else:
            if not os.path.exists(os.path.join(path, str(frame))):
                os.makedirs(os.path.join(path, str(frame)))
            torch.save(model.state_dict(), os.path.join(path, str(frame), name))

    def evaluate(self, policy: Callable, repeat: int = 1) -> float:
        """Evaluation of a policy on the environment

        Args:
            policy (Callable): Function from observations to actions
            repeat (int, optional): Number of repeats. Defaults to 1.

        Returns:
            float: Mean reward
        """
        evaluation_reward, evaluation_maximum_reward, evaluation_episode_length = evaluate(self.env, policy=policy, repeat=repeat)
        self.log(
            ({
                "evaluation/reward": evaluation_reward,
                "evaluation/max_reward": evaluation_maximum_reward,
                "evaluation/episode_length": evaluation_episode_length
            })
        )
        return evaluation_reward

    def log(self, kwargs):
        """ Logging a metric

        Args:
            kwargs (_type_): Passed to Wandb
        """
        if not self.wandb_enable:
            warnings.warn("Logging is not yet implemented without wandb")
            return
        wandb.log(kwargs)

    def close(self):
        """ Release resources
        """
        self.env.close()
        if self.wandb_enable:
            wandb.finish()
    
    def clear(self):
        """ Deletes associated files
        """
        shutil.rmtree(self.path)

    def __getattr__(self, name: str):
        return self.env.__getattribute__(name)

def evaluate(env: Env, policy: Callable, repeat: int = 1):
    """ Evaluates an episode

    Args:
        env (Env): A Gym environment 
        policy (Callable): Function which accepts the observation from gym, returning an action within the action space
        repeat (int, optional): Number of attempts. Defaults to 1.
    """
    total_reward = 0
    maximum_reward = -math.inf
    episode_length = 0
    for _ in range(repeat):
        episode_reward = 0
        with torch.no_grad():
            env.close()
            obs, _ = env.reset()
            terminated = False
            truncated = False
            while not terminated and not truncated:
                episode_length += 1
                obs, reward, terminated, truncated, _ = env.step(policy(obs))
                total_reward += reward
                episode_reward += reward
        maximum_reward = max(episode_reward, maximum_reward)
    # Calculate reward 
    total_reward = total_reward / repeat
    episode_length = episode_length / repeat
    return total_reward, maximum_reward, episode_length
