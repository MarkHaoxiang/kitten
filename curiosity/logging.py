import copy
import warnings
import os
from typing import Callable, Dict, Optional

import torch
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
                 wandb_enable: bool = False):
        super().__init__()
        # State
        self.project_name = project_name
        self.wandb_enable = wandb_enable
        self.config = config
        self.env = copy.deepcopy(env)
        self.path = os.path.join(path, project_name)
        # Log
        os.makedirs(self.path)
        os.makedirs(self.path, "checkpoint")

        if video:
            if env.render_mode != "rgb_array":
                raise ValueError(f"Video mode requires rgb_array render mode but got {env.render_mode}")
            self.env = RecordVideo(
                self.env,
                os.path.join(self.path, "video"),
                episode_trigger=lambda x: x%video == 0,
                disable_logger=True
            )

        if self.wandb_enable:
            wandb.init(
                project = "curiosity",
                name = self.project_name,
                config = self.config
            )

    def evaluate(self, policy: Callable, repeat: int = 1) -> float:
        """Evaluation of a policy on the environment

        Args:
            policy (Callable): Function from observations to actions
            repeat (int, optional): Number of repeats. Defaults to 1.

        Returns:
            float: Mean reward
        """
        evaluation_reward = evaluate(self.env, policy=policy, repeat=repeat)
        self.log(
            evaluation_reward({
                "evaluation/reward": evaluation_reward
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
    for _ in range(repeat):
        with torch.no_grad():
            env.close()
            obs, _ = env.reset()
            terminated = False
            truncated = False
            while not terminated and not truncated:
                obs, reward, terminated, truncated, _ = env.step(policy(obs))
                total_reward += reward   
    # Calculate reward 
    total_reward = total_reward / repeat
    return total_reward
