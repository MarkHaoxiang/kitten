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
from curiosity.world import IntrinsicCuriosityModule

parser = argparse.ArgumentParser(
    prog="TD3+ICM",
    description= """Fujimoto, et al. Addressing Function Approximation Error in Actor-Critic Methods. 2018. Adds clipped double critic, delayed policy updates, and value function smoothing  to DDPG. Pathak, Deepak, et al. Curiosity-Driven Exploration by Self-Supervised Prediction. 2017. Adds intrinsic curiosity, a type of exploration world model."""
)
parser.add_argument(
    '-c',
    "--config",
    default="config/td3.json",
    required=False,
    help="The configuration file to pass in."
)
args = parser.parse_args()

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
          target_noise: float,
          target_noise_clip: float,
          features: int,
          critic_update_frequency: int,
          policy_update_frequency: float,
          replay_buffer_capacity: int,
          total_frames: int,
          minibatch_size: int,
          initial_collection_size: int,
          tau: float,
          lr: float,
          icm_encoding_features: int,
          eta: float,
          beta: float,
          **kwargs):
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
        target_noise (float): Smoothing noise standard deviation added to policy for the critic update
        target_noise_clip (float): Clip target noise
        features (int): Helps define the complexity of neural nets used
        critic_update_frequency (int): Frames between each critic update
        policy_update_frequency (float): Ratio of critic updates - actor updates
        replay_buffer_capacity (int): Capacity of replay buffer
        total_frames (int): Total frames to train on
        minibatch_size (int): Batch size to run gradient descent
        initial_collection_size (int): Early start to help with explorations by taking random actions
        tau (float): Speed at which target network follows main networks
        lr (float): Optimiser step size
        icm_encoding_features (float): Encoding size for intrinsic curiosity
        eta (float): Scale of intrinsic reward for intrinsic curiosity
        beta (float): Scale of forward/inverse loss for intrinsic curiosity

    Raises:
        ValueError: Invalid configuration passed
    """
    
    # Metadata
    PROJECT_NAME = "td3_{}_{}".format(environment, str(datetime.now()).replace(":","-").replace(".","-"))
    random.seed(seed)
    policy_update_frequency = int(policy_update_frequency * critic_update_frequency)


    # Create Environments
    env = AutoResetWrapper(gym.make(environment, render_mode="rgb_array"))
    env.reset(seed=seed)
    env_action_scale = torch.tensor(env.action_space.high-env.action_space.low, device=DEVICE) / 2.0

    # Define Actor / Critic
    actor = AddTargetNetwork(build_actor(env, features, exploration_factor), device=DEVICE)
    critic_1 = AddTargetNetwork(build_critic(env, features), device=DEVICE)
    critic_2 = AddTargetNetwork(build_critic(env, features), device=DEVICE)

    # Define curiosity
    icm = IntrinsicCuriosityModule.build(
        obs_size=env.observation_space.shape[0],
        encoding_size=icm_encoding_features,
        action_size=env.action_space.shape[0],
        eta=eta,
        beta=beta,
        device=DEVICE
    )

    # Initialise Replay Buffer    
    memory = build_replay_buffer(env, capacity=replay_buffer_capacity, device=DEVICE)

    # Loss
    loss_critic_fn = nn.MSELoss()
    optim_actor = torch.optim.Adam(params=actor.net.parameters(), lr=lr)
    optim_critic = torch.optim.Adam(
        params=list(critic_1.net.parameters()) + list(critic_2.net.parameters()),
        lr=lr
    )
    optim_icm = torch.optim.Adam(params=icm.parameters(), lr=lr)

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
        evaluator.checkpoint(critic_1.net, "critic_1")
        evaluator.checkpoint(critic_2.net, "critic_2")

    # Training loop
    obs, _ = env.reset()
    epoch = 0
    with tqdm(total=total_frames // frames_per_epoch, file=sys.stdout) as pbar:
        early_start(env, memory, initial_collection_size)
        # Logging
        loss_critic_value = 0
        loss_actor_value = 0
        loss_icm_value = 0

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

            # TD3 Update
            s_t0, a_t, r_e, s_t1, d = memory.sample(minibatch_size, continuous=False)
                # Add intrinsic reward
            r_i = icm(s_t0, s_t1, a_t)
            r_t = r_e + r_i

            if step % critic_update_frequency == 0:
                # Critic
                x1 = critic_1(torch.cat((s_t0, a_t), 1)).squeeze()
                x2 = critic_2(torch.cat((s_t0, a_t), 1)).squeeze()
                with torch.no_grad():
                    next_action = actor.target(s_t1)
                    next_action += torch.clamp(
                        torch.normal(0, target_noise, next_action.shape, device=DEVICE),
                        min = -target_noise_clip,
                        max = target_noise_clip,
                    ) * env_action_scale
                    next_action = torch.clamp(
                        next_action,
                        min=torch.tensor(env.action_space.low,dtype=torch.float32, device=DEVICE),
                        max=torch.tensor(env.action_space.low,dtype=torch.float32, device=DEVICE)
                    )
                    target_max_1 = critic_1.target(torch.cat((s_t1, next_action), 1)).squeeze()
                    target_max_2 = critic_2.target(torch.cat((s_t1, next_action), 1)).squeeze()
                    y = (r_t + (~d) * torch.minimum(target_max_1, target_max_2) * discount_factor)
                loss_critic = loss_critic_fn(y, x1) + loss_critic_fn(y, x2)
                loss_critic_value = loss_critic.item()
                optim_critic.zero_grad()
                loss_critic.backward()
                optim_critic.step()

            if step % policy_update_frequency == 0:
                # Actor
                desired_action = actor(s_t0)
                loss_actor = -critic_1(torch.cat((s_t0, desired_action), 1)).mean()
                optim_actor.zero_grad()
                loss_actor_value = loss_actor.item()
                loss_actor.backward()
                optim_actor.step()

                # Update Target Networks
                critic_1.update_target_network(tau)
                critic_2.update_target_network(tau)
                actor.update_target_network(tau)

                # Update icm
                loss_icm = icm.loss(s_t0, s_t1, a_t)
                optim_icm.zero_grad()
                loss_icm_value = loss_icm.item()
                loss_icm.backward()
                optim_icm.step()

            # Epoch Logging
            if step % frames_per_epoch == 0 :
                epoch += 1
                reward = evaluator.evaluate(
                    policy = lambda x: actor(torch.tensor(x, device=DEVICE, dtype=torch.float32)).cpu().numpy(),
                    repeat=eval_repeat
                )
                critic_value = critic_1(torch.cat((evaluator.saved_reset_states, actor(evaluator.saved_reset_states)), 1)).mean().item()
                evaluator.log({
                    "train/frame": step,
                    "train/critic_loss": loss_critic_value,
                    "train/actor_loss": loss_actor_value,
                    "train/critic_value": critic_value,
                    "train/icm_loss": loss_icm_value,
                    "train/intrinsic_reward": torch.mean(r_i),
                    "train/extrinsic_reward": torch.mean(r_e)
                })
                pbar.set_description(f"epoch {epoch} reward {reward} critic loss {loss_critic_value} actor loss {loss_actor_value}")
                pbar.update(1)
            
            if checkpoint and step % frames_per_checkpoint == 0:
                evaluator.checkpoint(actor.net, "actor", step)
                evaluator.checkpoint(critic_1.net, "critic_1", step)
                evaluator.checkpoint(critic_2.net, "critic_2", step)

        evaluator.close()
        env.close()

if __name__ == "__main__":
    with open(args.config, 'r') as f:
        config = json.load(f)
    train(config, **config)
