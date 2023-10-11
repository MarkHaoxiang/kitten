import argparse
import copy
from datetime import datetime
import json
import random
import sys
from typing import Dict

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
parser = argparse.ArgumentParser(
    prog="TD3",
    description= "Fujimoto, et al. Addressing Function Approximation Error in Actor-Critic Methods. 2018."
)
parser.add_argument("config", help="The configuration file to pass in.")
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(config: Dict,
          seed: int,
          frames_per_epoch: int,
          frames_per_video: int,
          eval_repeat: int,
          wandb: bool,
          environment: str,
          discount_factor: float,
          exploration_factor: float,
          features: int,
          critic_update_frequency: int,
          policy_update_frequency: float,
          replay_buffer_capacity: int,
          total_frames: int,
          minibatch_size: int,
          initial_collection_size: int,
          tau: float,
          lr: float):
    """ Train TD3

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
        critic_update_frequency (int): Frames between each critic update
        policy_update_frequency (float): Ratio of critic updates - actor updates
        replay_buffer_capacity (int): Capacity of replay buffer
        total_frames (int): Total frames to train on
        minibatch_size (int): Batch size to run gradient descent
        initial_collection_size (int): Early start to help with explorations by taking random actions
        tau (float): Speed at which target network follows main networks
        lr (float): Optimiser step size

    Raises:
        ValueError: Invalid configuration passed
    """
    
    # Metadata
    PROJECT_NAME = "td3_{}_{}".format(environment, str(datetime.now()).replace(":","-").replace(".","-"))
    random.seed(seed)
    policy_update_frequency = int(policy_update_frequency * critic_update_frequency)

    # Create Environments
    env = AutoResetWrapper(gym.make(environment, render_mode="rgb_array"))
    if not isinstance(env.action_space, Box) or len(env.action_space.shape) != 1:
        raise ValueError("This implementation of TD3 requires 1D continuous actions")
    env.reset(seed=seed)

    # Define Actor / Critic
    actor, critic_1 = build_actor_critic(env, features, exploration_factor)
    actor, critic_1 = AddTargetNetwork(actor, device=DEVICE), AddTargetNetwork(critic_1, device=DEVICE)
    critic_2 = copy.deepcopy(critic_1)

    # Initialise Replay Buffer    
    memory = build_replay_buffer(env, capacity=replay_buffer_capacity, device=DEVICE)

    # Loss
    loss_critic_fn = nn.MSELoss()
    optim_actor = torch.optim.Adam(params=actor.net.parameters(), lr=lr)
    optim_critic = torch.optim.Adam(params=critic_1.net.parameters(), lr=lr)

    # Logging
    evaluator = EvaluationEnv(
        env=env,
        project_name=PROJECT_NAME,
        config=config,
        video=eval_repeat*frames_per_video//frames_per_epoch,
        wandb_enable=wandb
    )
    evaluator.reset(seed=seed)

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
                action = actor(torch.tensor(obs, device=DEVICE), noise=True) \
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

            epoch_update = step % frames_per_epoch == 0 
            # DDPG Update
                # Critic
            s_t0, a_t, r_t, s_t1, d = memory.sample(minibatch_size, continuous=False)
            x = critic_1(torch.cat((s_t0, a_t), 1)).squeeze()
            with torch.no_grad():
                target_max = (~d) * critic_1.target(torch.cat((s_t1, actor.target(s_t1)), 1)).squeeze()
                y = (r_t + target_max * discount_factor)
            loss_critic = loss_critic_fn(y, x)
            optim_critic.zero_grad()
            loss_critic_value = loss_critic.item()
            loss_critic.backward()
            optim_critic.step()
                # Actor
            desired_action = actor(s_t0)
            loss_actor = -critic_1(torch.cat((s_t0, desired_action), 1)).mean()
            optim_actor.zero_grad()
            loss_actor_value = loss_actor.item()
            loss_actor.backward()
            optim_actor.step()
            critic_1.update_target_network(tau)
            actor.update_target_network(tau)

            # Epoch Logging
            if epoch_update:
                epoch += 1
                reward = evaluator.evaluate(
                    policy = lambda x: actor(torch.tensor(x, device=DEVICE)).cpu().numpy(),
                    repeat=eval_repeat
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
    with open(args.config, 'r') as f:
        config = json.load(f)
    train(config, **config)
