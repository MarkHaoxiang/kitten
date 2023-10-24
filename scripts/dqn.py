import argparse
from datetime import datetime
import json
import random
import sys
from typing import Dict

import gymnasium as gym
from gymnasium.wrappers.autoreset import AutoResetWrapper
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.spaces import Discrete
import torch
import torch.nn as nn
from tqdm import tqdm
from curiosity.experience import (
    build_replay_buffer,
    early_start
)
from curiosity.exploration import epsilon_greedy
from curiosity.logging import CuriosityArgumentParser, EvaluationEnv
from curiosity.nn import (
    AddTargetNetwork,
    build_critic
)
parser = CuriosityArgumentParser(
    prog="DQN",
    description="V Mnih, et al. Playing Atari with Deep Reinforcement Learning. 2013."
)
# An implementation of DQN
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(config: Dict,
          seed: int = 0,
          frames_per_epoch: int = 1000,
          frames_per_video: int = 50000,
          eval_repeat: int = 10,
          wandb: bool = False,
          environment: str = "Acrobot-v1",
          atari: bool = False,
          discount_factor: float = 0.99,
          critic_features: int = 128,
          target_update_frequency: int = 500,
          replay_buffer_capacity: int = 10000,
          total_frames: int = 500000,
          minibatch_size: int = 128,
          initial_collection_size: int = 10000,
          update_frequency: int = 10,
          initial_exploration_factor: float = 1.0,
          final_exploration_factor: float = 0.05,
          exploration_anneal_time: int = 50000):
    """ Train DQN

    Args:
            # Metadata
        config (Dict): Dictionary containing all arguments
            # Experimentation and Logging
        seed (int, optional): Random seed used to setup experiment. Defaults to 0.
        frames_per_epoch (int, optional): Number of frames between each logging point. Defaults to 1000.
        frames_per_video (int, optional): Number f frames between each video. Defaults to 50000.
        eval_repeat (int, optional): Number of episodes to run in an evaluation. Defaults to 10.
        wandb (bool, optional): Toggle to log to Weights&Biases. Defaults to False.
            # Environment
        environment (str, optional): Environment name in registry. Defaults to "Acrobot-v1".
        atari (bool, optional): Toggle to indicate an Atari environment with pixel features. Defaults to False.
        discount_factor (float, optional): Reward discount. Defaults to 0.99.
            # Algorithm
        critic_features (int, optional): Helps define the complexity of the critic neural net. Defaults to 128.
        target_update_frequency (int, optional): Steps between each update to the target network. Defaults to 500.
        replay_buffer_capacity (int, optional): Capacity of the replay buffer. Defaults to 10000.
        total_frames (int, optional): Total frames to train on. Defaults to 500000.
        minibatch_size (int, optional): Batch size to run gradient descent. Defaults to 128.
        initial_collection_size (int, optional): Early start to help exploration by taking random actions. Defaults to 10000.
        update_frequency (int, optional): Steps between each network update. Defaults to 10.
        initial_exploration_factor (float, optional): Exploration factor, initial. Defaults to 1.0.
        final_exploration_factor (float, optional): Exploration factor, final. Defaults to 0.05.
        exploration_anneal_time (int, optional): Time taken to anneal betwen initial -> final exploration factor. Defaults to 50000.

    Raises:
        ValueError: Invalid configuration passed.
    """
    PROJECT_NAME = parser.generate_project_name(environment, "dqn")
    random.seed(seed)
    torch.manual_seed(seed)

    # Create Environments
    if atari:
        env = gym.make(environment, render_mode="rgb_array", frameskip=1)
        env = AtariPreprocessing(env)
    else:
        env = gym.make(environment, render_mode="rgb_array")
    env = AutoResetWrapper(env)    
    if not isinstance(env.action_space, Discrete):
        raise ValueError("DQN requires discrete actions")
    env.reset(seed=seed)

    # Define Critic
    critic = AddTargetNetwork(build_critic(env, critic_features), device=DEVICE)

    # Initialise Replay Buffer
    memory = build_replay_buffer(env, capacity=replay_buffer_capacity, device=DEVICE)

    # Loss
    loss = nn.MSELoss()
    optim = torch.optim.Adam(params=critic.net.parameters())

    # Logging
    evaluator = EvaluationEnv(
        env=env,
        project_name=PROJECT_NAME,
        config=config,
        video=eval_repeat*frames_per_video//frames_per_epoch,
        wandb_enable=wandb
    )
    evaluator.reset(seed=seed)

    # ====================
    # Training loop
    # ====================
    obs, _ = env.reset()
    epoch = 0
    with tqdm(total=total_frames // frames_per_epoch, file=sys.stdout) as pbar:
        early_start(env, memory, initial_collection_size)
        # Logging
        for step in range(total_frames):

            # ====================
            # Data Collection
            # ====================
            train_step = max(0, step-initial_collection_size)
            exploration_stage = min(1, train_step / exploration_anneal_time)
            epsilon = exploration_stage * final_exploration_factor + (1-exploration_stage) * initial_exploration_factor
            # Calculate action
            action = epsilon_greedy(torch.argmax(critic(torch.tensor(obs, device=DEVICE, dtype=torch.float32))), env.action_space, epsilon)
            # Step environment
            n_obs, reward, terminated, truncated, _ = env.step(action)
            # Update memory buffer
                # (s_t, a_t, r_t, s_t+1)
            transition_tuple = (obs, action, reward, n_obs, terminated)
            obs = n_obs
            memory.append(transition_tuple)

            # ====================
            # Gradient Update 
            # Mostly Update this
            # ====================

            epoch_update = step % frames_per_epoch == 0 
            if step % update_frequency == 0:
                # DQN Update
                s_t0, a_t, r_t, s_t1, d = memory.sample(minibatch_size, continuous=False)
                x = critic(s_t0)[torch.arange(minibatch_size), a_t]
                with torch.no_grad():
                    target_max = (~d) * torch.max(critic.target(s_t1), dim=1).values
                    y = (r_t + target_max * discount_factor)
                output = loss(y, x)
                if epoch_update:
                    critic_loss = output.item()
                optim.zero_grad()
                output.backward()
                optim.step()

            # Update target networks
            if step % target_update_frequency == 0:
                critic.update_target_network()

            # Epoch Logging
            if epoch_update:
                epoch += 1
                reward = evaluator.evaluate(policy = lambda x: torch.argmax(critic(torch.tensor(x, device=DEVICE, dtype=torch.float32))).cpu().numpy(), repeat=eval_repeat)
                pbar.set_description(f"epoch {epoch} reward {reward} critic loss {critic_loss} exploration factor {epsilon}")
                pbar.update(1)
                evaluator.log({
                    "train/frame": step,
                    "train/critic_loss": critic_loss,
                    "train/exploration_factor": epsilon
                })

        evaluator.close()
        env.close()

if __name__ == "__main__":
    parser.run(train)