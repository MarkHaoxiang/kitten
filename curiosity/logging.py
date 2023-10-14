import copy
import math
import warnings
import os
from typing import Callable, Dict, Optional
import json

import torch
import torch.nn as nn
from gymnasium import Env
from gymnasium.wrappers.record_video import RecordVideo
import wandb

class EvaluationEnv(Env):
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
        self.saved_reset_states = torch.tensor(self.saved_reset_states, device=device)

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
            dir = os.path.join(self.path, "wandb"),
            mode = "online" if wandb_enable else "offline"
        )

    def checkpoint(self, model: nn.Module, name: str, frame: Optional[int] = None):
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
        self.env.close()
        if self.wandb_enable:
            wandb.finish()

    def __getattr__(self, name: str):
        return self.env.__getattr__(name)

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
