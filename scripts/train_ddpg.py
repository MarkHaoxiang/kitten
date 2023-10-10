from datetime import datetime
import random
import sys
from typing import Dict

import numpy as np
import gymnasium as gym
from gymnasium.wrappers.autoreset import AutoResetWrapper
from gymnasium.spaces import Box
import torch
import torch.nn as nn
from tqdm import tqdm

from curiosity.experience import (
    build_replay_buffer,
    early_start
)
from curiosity.logging import EvaluationEnv 
from curiosity.nn import (
    ClassicalGaussianActor,
    AddTargetNetwork
)

# An implementation of DDPG
# Lillicrap, et al. Continuous control with deep reinforcement learning. 2015.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WANDB = False

default_config = {
    # Experimentation and Logging
    "seed": 0,
    "frames_per_epoch": 1000,
    "frames_per_video": 50000,
    "eval_repeat": 10,
    # Environemnt
    "environment":"Pendulum-v1",
    "discount_factor": 0.99,
    "exploration_factor": 0.1,
    # Network
    "features": 128,
    "target_update_frequency": 500,
    # Replay Buffer Capacity
    "replay_buffer_capacity": 10000,
    # Training
    "total_frames": 500000,
    "minibatch_size": 128,
    "initial_collection_size": 10000,
    "tau": 0.005,
    "lr": 0.001
}

def train(config: Dict = default_config):
    PROJECT_NAME = "dqn_{}_{}".format(config["environment"], str(datetime.now()).replace(":","-").replace(".","-"))
    random.seed(config['seed'])

    # Create Environments
    env = AutoResetWrapper(gym.make(config["environment"], render_mode="rgb_array"))
    if not isinstance(env.action_space, Box) or len(env.action_space.shape) != 1:
        raise ValueError("This implementation of DDPG requires 1D continuous actions")
    env.reset(seed=config['seed'])

    # Define Critic
    actor, critic = build_actor_critic(env, config["features"], config["exploration_factor"])
    actor, critic = AddTargetNetwork(actor, device=DEVICE), AddTargetNetwork(critic, device=DEVICE)

    # Initialise Replay Buffer    
    memory = build_replay_buffer(env, capacity=config["replay_buffer_capacity"], device=DEVICE)

    # Loss
    loss_critic_fn = nn.MSELoss()
    optim_actor = torch.optim.Adam(params=actor.net.parameters(), lr=config["lr"])
    optim_critic = torch.optim.Adam(params=critic.net.parameters(), lr=config["lr"])

    # Logging
    evaluator = EvaluationEnv(
        env=env,
        project_name=PROJECT_NAME,
        config=config,
        video=config["eval_repeat"]*config["frames_per_video"]//config["frames_per_epoch"],
        wandb_enable=WANDB
    )
    evaluator.reset(seed=config["seed"])

    # Training loop
    obs, _ = env.reset()
    epoch = 0
    with tqdm(total=config["total_frames"] // config["frames_per_epoch"], file=sys.stdout) as pbar:
        early_start(env, memory, config["initial_collection_size"])
        # Logging
        loss_critic_value = 0
        loss_actor_value = 0

        for step in range(config["total_frames"]):
            # ====================
            # Data Collection
            # ====================

            # Calculate action
            with torch.no_grad():
                action = actor(torch.tensor(obs, device=DEVICE), noise=True) \
                    .cpu() \
                    .detach() \
                    .numpy() \
                    .clip(env.action_space.low, env.action_space.high)
            # Step environment
            n_obs, reward, terminated, truncated, infos = env.step(action)
            # Update memory buffer
                # (s_t, a_t, r_t, s_t+1, d)
            transition_tuple = (obs, action, reward, n_obs, terminated)
            obs = n_obs
            memory.append(transition_tuple)

            # ====================
            # Gradient Update 
            # Mostly Update this
            # ====================

            epoch_update = step % config["frames_per_epoch"] == 0 
            # DDPG Update
                # Critic
            s_t0, a_t, r_t, s_t1, d = memory.sample(config["minibatch_size"], continuous=False)
            x = critic(torch.cat((s_t0, a_t), 1)).squeeze()
            with torch.no_grad():
                target_max = (~d) * critic.target(torch.cat((s_t1, actor.target(s_t1)), 1)).squeeze()
                y = (r_t + target_max * config["discount_factor"])
            loss_critic = loss_critic_fn(y, x)
            optim_critic.zero_grad()
            loss_critic_value = loss_critic.item()
            loss_critic.backward()
            optim_critic.step()
                # Actor
            desired_action = actor(s_t0)
            loss_actor = -critic(torch.cat((s_t0, desired_action), 1)).mean()
            optim_actor.zero_grad()
            loss_actor_value = loss_actor.item()
            loss_actor.backward()
            optim_actor.step()
            critic.update_target_network(config["tau"])
            actor.update_target_network(config["tau"])

            # Epoch Logging
            if epoch_update:
                epoch += 1
                reward = evaluator.evaluate(
                    policy = lambda x: actor(torch.tensor(x, device=DEVICE)).cpu().numpy(),
                    repeat=config["eval_repeat"]
                )
                pbar.set_description(f"epoch {epoch} reward {reward} critic loss {loss_critic_value} actor loss {loss_actor_value}")
                pbar.update(1)
                evaluator.log({
                    "frame": step,
                    "eval_reward": reward,
                    "critic_loss": loss_critic_value,
                    "actor_loss": loss_actor_value,
                })

        evaluator.close()
        env.close()

def build_actor_critic(env: gym.Env, N: int = 128, exploration_factor: float = 0.1):
    actor = ClassicalGaussianActor(env, N, exploration_factor).to(device=DEVICE)
    critic = nn.Sequential(
        nn.LazyLinear(out_features=N),
        nn.Tanh(),
        nn.Linear(in_features=N, out_features=N),
        nn.Tanh(),
        nn.Linear(in_features=N, out_features=1)
    )

    return actor, critic

if __name__ == "__main__":
    train()