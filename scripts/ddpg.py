import argparse
from datetime import datetime
import json
import random
import sys
from typing import Dict

import gymnasium as gym
from gymnasium.wrappers.autoreset import AutoResetWrapper
import torch
import torch.nn as nn
from tqdm import tqdm

from curiosity.experience import (
    build_replay_buffer,
    early_start
)
from curiosity.logging import EvaluationEnv 
from curiosity.nn import (
    AddTargetNetwork,
    build_actor,
    build_critic
)

parser = argparse.ArgumentParser(
    prog="DDPG",
    description= "Lillicrap, et al. Continuous control with deep reinforcement learning. 2015."
)
parser.add_argument(
    '-c',
    "--config",
    default="scripts/ddpg.json",
    required=False,
    help="The configuration file to pass in."
)
args = parser.parse_args()


# An implementation of DDPG
# Lillicrap, et al. Continuous control with deep reinforcement learning. 2015.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(config: Dict,
          seed: int,
          frames_per_epoch: int,
          frames_per_video: int,
          eval_repeat: int,
          wandb: bool,
          frames_per_checkpoint: int,
          environment: str,
          discount_factor: float,
          exploration_factor: float,
          features: int,
          target_update_frequency: int,
          replay_buffer_capacity: int,
          total_frames: int,
          minibatch_size: int,
          initial_collection_size: int,
          tau: float,
          lr: float,
          **kwargs):
    """ Train DDPG

    Args:
            # Metadata
        config (Dict): Dictionary containing all arguments
            # Experimentation and Logging
        seed (int): random seed used to setup experiment
        frames_per_epoch (int): Number of frames between each logging point
        frames_per_video (int): Number of frames between each video
        eval_repeat (int): Number of episodes to run in an evaluation
        wandb (bool): Toggle to log to Weights&Biases
            # Environment
        environment (str): Environment name in registry
        discount_factor (float): Reward discount
            # Algorithm
        exploration_factor (float): Gaussian noise standard deviation for exploration
        features (int): Helps define the complexity of neural nets used
        target_update_frequency (int): Frames between each network update
        e
        
        fer_capacity (int): Capacity of replay buffer
        total_frames (int): Total frames to train on
        minibatch_size (int): Batch size to run gradient descent
        initial_collection_size (int): Early start to help with explorations by taking random actions
        tau (float): Speed at which target network follows main networks
        lr (float): Optimiser step size

    Raises:
        ValueError: Invalid configuration passed
    """
    # Metadata
    PROJECT_NAME = "ddpg_{}_{}".format(environment, str(datetime.now()).replace(":","-").replace(".","-"))
    random.seed(seed)

    # Create Environments
    env = AutoResetWrapper(gym.make(environment, render_mode="rgb_array"))
    env.reset(seed=seed)

    # Define Actor / Critic
    actor = AddTargetNetwork(build_actor(env, features, exploration_factor), device=DEVICE)
    critic = AddTargetNetwork(build_critic(env, features), device=DEVICE)

    # Initialise Replay Buffer    
    memory = build_replay_buffer(env, capacity=replay_buffer_capacity, device=DEVICE)

    # Loss
    loss_critic_fn = nn.MSELoss()
    optim_actor = torch.optim.Adam(params=actor.net.parameters(), lr=lr)
    optim_critic = torch.optim.Adam(params=critic.net.parameters(), lr=lr)

    # Logging
    evaluator = EvaluationEnv(
        env=env,
        project_name=PROJECT_NAME,
        config=config,
        video=eval_repeat*frames_per_video//frames_per_epoch if frames_per_video > 0 else None,
        wandb_enable=wandb,
        device=DEVICE
    )
    evaluator.reset(seed=seed)

    checkpoint = frames_per_checkpoint > 0
    if checkpoint:
        evaluator.checkpoint(actor.net, "actor")
        evaluator.checkpoint(critic.net, "critic")

    # Training loop
    obs, _ = env.reset()
    epoch = 0
    with tqdm(total=total_frames // frames_per_epoch, file=sys.stdout) as pbar:
        early_start(env, memory, initial_collection_size)
        # Logging
        loss_critic_value = 0
        loss_actor_value = 0

        for step in range(total_frames):
            # ====================
            # Data Collection
            # ====================

            # Calculate action
            with torch.no_grad():
                action = actor(torch.tensor(obs, device=DEVICE, dtype=torch.float32), noise=True) \
                    .cpu().detach().numpy() \
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

            # DDPG Update
                # Critic
            if step % target_update_frequency == 0:
                s_t0, a_t, r_t, s_t1, d = memory.sample(minibatch_size, continuous=False)
                x = critic(torch.cat((s_t0, a_t), 1)).squeeze()
                with torch.no_grad():
                    target_max = (~d) * critic.target(torch.cat((s_t1, actor.target(s_t1)), 1)).squeeze()
                    y = (r_t + target_max * discount_factor)
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
                critic.update_target_network(tau)
                actor.update_target_network(tau)

            # Epoch Logging
            if step % frames_per_epoch == 0:
                epoch += 1
                reward = evaluator.evaluate(
                    policy = lambda x: actor(torch.tensor(x, device=DEVICE, dtype=torch.float32)).cpu().numpy(),
                    repeat=eval_repeat
                )
                critic_value =  critic(torch.cat((evaluator.saved_reset_states, actor(evaluator.saved_reset_states)), 1)).mean().item()
                evaluator.log({
                    "train/frame": step,
                    "train/critic_loss": loss_critic_value,
                    "train/actor_loss": loss_actor_value,
                    "train/critic_value": critic_value 
                })
                pbar.set_description(f"epoch {epoch} reward {reward} critic loss {loss_critic_value} actor loss {loss_actor_value}")
                pbar.update(1)

            if checkpoint and step % frames_per_checkpoint == 0:
                evaluator.checkpoint(actor.net, "actor", step)
                evaluator.checkpoint(critic.net, "critic", step)

        evaluator.close()
        env.close()

if __name__ == "__main__":
    with open(args.config, 'r') as f:
        config = json.load(f)
    train(config, **config)
