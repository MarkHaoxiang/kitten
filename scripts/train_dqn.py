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
from curiosity.logging import EvaluationEnv
from curiosity.nn import (
    AddTargetNetwork,
    build_critic
)

parser = argparse.ArgumentParser(
    prog="DQN",
    description= "Mnih, Volodymyr, et al. Playing Atari with Deep Reinforcement Learning. 2013."
)
parser.add_argument("config", help="The configuration file to pass in.")
args = parser.parse_args()

# An implementation of DQN
# 
# With improvements to form RAINBOW (TODO)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(config: Dict):
    PROJECT_NAME = "dqn_{}_{}".format(config["environment"], str(datetime.now()).replace(":","-").replace(".","-"))
    random.seed(config['seed'])

    # Create Environments
    if config["atari"]:
        env = gym.make(config["environment"], render_mode="rgb_array", frameskip=1)
        env = AtariPreprocessing(env)
    else:
        env = gym.make(config["environment"], render_mode="rgb_array")
    env = AutoResetWrapper(env)    
    if not isinstance(env.action_space, Discrete):
        raise ValueError("DQN requires discrete actions")
    env.reset(seed=config["seed"])

    # Define Critic
    critic = AddTargetNetwork(build_critic(env, config["critic_features"]), device=DEVICE)

    # Initialise Replay Buffer
    memory = build_replay_buffer(env, capacity=config["replay_buffer_capacity"], device=DEVICE)

    # Loss
    loss = nn.MSELoss()
    optim = torch.optim.Adam(params=critic.net.parameters())

    # Logging
    evaluator = EvaluationEnv(
        env=env,
        project_name=PROJECT_NAME,
        config=config,
        video=config["eval_repeat"]*config["frames_per_video"]//config["frames_per_epoch"],
        wandb_enable=config["wandb"]
    )
    evaluator.reset(seed=config["seed"])

    # ====================
    # Training loop
    # ====================
    obs, _ = env.reset()
    epoch = 0
    with tqdm(total=config["total_frames"] // config["frames_per_epoch"], file=sys.stdout) as pbar:
        early_start(env, memory, config["initial_collection_size"])
        # Logging
        for step in range(config["total_frames"]):

            # ====================
            # Data Collection
            # ====================
            train_step = max(0, step-config["initial_collection_size"])
            exploration_stage = min(1, train_step / config["exploration_anneal_time"])
            epsilon = exploration_stage * config["final_exploration_factor"] + (1-exploration_stage) * config["initial_exploration_factor"]
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

            epoch_update = step % config["frames_per_epoch"] == 0 
            if step % config["update_frequency"] == 0:
                # DQN Update
                s_t0, a_t, r_t, s_t1, d = memory.sample(config["minibatch_size"], continuous=False)
                x = critic(s_t0)[torch.arange(config["minibatch_size"]), a_t]
                with torch.no_grad():
                    target_max = (~d) * torch.max(critic.target(s_t1), dim=1).values
                    y = (r_t + target_max * config["discount_factor"])
                output = loss(y, x)
                if epoch_update:
                    critic_loss = output.item()
                optim.zero_grad()
                output.backward()
                optim.step()

            # Update target networks
            if step % config["target_update_frequency"] == 0:
                critic.update_target_network()

            # Epoch Logging
            if epoch_update:
                epoch += 1
                reward = evaluator.evaluate(policy = lambda x: torch.argmax(critic(torch.tensor(x, device=DEVICE, dtype=torch.float32))).cpu().numpy(), repeat=config["eval_repeat"])
                pbar.set_description(f"epoch {epoch} reward {reward} critic loss {critic_loss} exploration factor {epsilon}")
                pbar.update(1)
                evaluator.log({
                    "frame": step,
                    "eval_reward": reward,
                    "critic_loss": critic_loss,
                    "exploration_factor": epsilon
                })

        evaluator.close()
        env.close()

if __name__ == "__main__":
    with open(args.config, 'r') as f:
        config = json.load(f)
    train(config)