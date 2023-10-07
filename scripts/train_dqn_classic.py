from datetime import datetime
import random
import sys
from typing import Dict

import gymnasium as gym
from gymnasium.wrappers.autoreset import AutoResetWrapper
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.spaces import Box, Discrete
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from curiosity.experience import ReplayBuffer
from curiosity.exploration import epsilon_greedy
from curiosity.logging import evaluate
from curiosity.nn import AddTargetNetwork

# An implementation of DQN
# Mnih, Volodymyr, et al. Playing Atari with Deep Reinforcement Learning. 2013.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WANDB = False

default_config = {
    # Experimentation and Logging
    "seed": 0,
    "frames_per_epoch": 1000,
    "frames_per_video": 50000,
    "eval_repeat": 10,
    # Environemnt
    "environment": "CartPole-v1",
    "frame_skip": 1,
    "discount_factor": 0.99,
    # Critic
    "critic_features": 128,
    "target_update_frequency": 500,
    # Replay Buffer Capacity
    "replay_buffer_capacity": 10000,
    # Training
    "total_frames": 500000,
    "minibatch_size": 128,
    "initial_collection_size": 10000,
    "update_frequency": 10,
    # Exploration
        # Epsilon Greedy Exploration
    "initial_exploration_factor": 1,
    "final_exploration_factor": 0.05,
    "exploration_anneal_time": 50000
}

def train(config: Dict = default_config):

    PROJECT_NAME = "dqn_{}_{}".format(config["environment"], str(datetime.now()).replace(":","-").replace(".","-"))
    random.seed(config['seed'])

    # Create Environments
    env_train = AutoResetWrapper(gym.make(config["environment"]))
    env_eval = RecordVideo(
        gym.make(config["environment"], render_mode="rgb_array"),
        f"log/video/{PROJECT_NAME}",
        episode_trigger=lambda x: x% (config["eval_repeat"]*config["frames_per_video"]//config["frames_per_epoch"])==0,
        disable_logger=True
    )
    if not isinstance(env_train.action_space, Discrete):
        raise ValueError("DQN requires discrete actions")
    if not isinstance(env_train.observation_space, Box) or len(env_train.observation_space.shape) != 1:
        raise ValueError("Critic is defined only for 1D box observations. Is this a pixel space?")
    env_train.reset(seed=config['seed'])
    env_eval.reset(seed=config['seed'])

    # Define Critic
    N = config['critic_features']
    critic = \
    AddTargetNetwork(nn.Sequential(
        nn.LazyLinear(out_features=N),
        nn.Tanh(),
        nn.Linear(in_features=N, out_features=N),
        nn.Tanh(),
        nn.Linear(in_features=N, out_features=env_train.action_space.n)
    ), device = DEVICE)

    # Initialise Replay Buffer
    memory = ReplayBuffer(
        capacity=config["replay_buffer_capacity"],
        shape=(env_train.observation_space.shape, env_train.action_space.shape, 1, env_train.observation_space.shape, 1),
        dtype=(torch.float32, torch.int, torch.float32, torch.float32, torch.bool),
        device=DEVICE
    )
    # Loss
    loss = nn.MSELoss()
    optim = torch.optim.Adam(params=critic.parameters())

    # Logging
    if WANDB:
        wandb.init(
            project = "curiosity",
            name = PROJECT_NAME,
            config = config
        )

    # Training loop
    obs, _ = env_train.reset()
    epoch = 0
    with tqdm(total=config["total_frames"] // config["frames_per_epoch"], file=sys.stdout) as pbar:
        # Logging
        critic_loss = 0
        for step in range(config["total_frames"] // config["frame_skip"] + 1):

            # ====================
            # Data Collection
            # ====================

            train_step = max(0, step-config["initial_collection_size"])
            exploration_stage = min(1, train_step / config["exploration_anneal_time"])
            epsilon = exploration_stage * config["final_exploration_factor"] + (1-exploration_stage) * config["initial_exploration_factor"]
            # Calculate action
            action = epsilon_greedy(torch.argmax(critic(torch.Tensor(obs).to(device=DEVICE))), env_train.action_space, epsilon)
            # Step environment
            reward = 0
            for _ in range(config["frame_skip"]):
                n_obs, n_reward, n_terminated, n_truncated, _ = env_train.step(action)
                done = n_terminated or n_truncated
                reward += n_reward
                if done: # New episode
                    break
            # Update memory buffer
                # (s_t, a_t, r_t, s_t+1)
            transition_tuple = (obs, action, [reward], n_obs, [n_terminated])
            obs = n_obs
            memory.append(transition_tuple)

            # Keep populating data
            if step < config["initial_collection_size"]:
                continue
            
            # ====================
            # Gradient Update 
            # Mostly Update this
            # ====================

            epoch_update = step % config["frames_per_epoch"] == 0 
            if step % config["update_frequency"] == 0:
                # DQN Update
                s_t0, a_t, r_t, s_t1, d = memory.sample(config["minibatch_size"], continuous=False)
                x = critic(s_t0)[torch.arange(config["minibatch_size"]), a_t]
                with torch.no_grad():
                    target_max = (~d).squeeze() * torch.max(critic.target(s_t1), dim=1).values
                    y = (r_t.squeeze() + target_max * config["discount_factor"])
                output = loss(y, x)
                critic_loss = output.item()
                optim.zero_grad()
                output.backward()
                optim.step()

            # Update target networks
            if step % config["target_update_frequency"] == 0:
                critic.update_target_network(1.0)

            # Epoch Logging
            if epoch_update:
                epoch += 1
                reward = evaluate(env=env_eval, policy = lambda x: torch.argmax(critic(torch.tensor(x,device=DEVICE))).cpu().numpy(), repeat=config["eval_repeat"])

                pbar.set_description(f"epoch {epoch} reward {reward} critic loss {critic_loss} exploration factor {epsilon}")
                pbar.update(1)
                if WANDB:
                    wandb.log({
                        "frame": step,
                        "eval_reward": reward,
                        "critic_loss": critic_loss,
                        "exploration_factor": epsilon
                    })

        env_eval.close()
        env_train.close()
        if WANDB:
            wandb.finish()

if __name__ == "__main__":
    train()