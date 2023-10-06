import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers.autoreset import AutoResetWrapper
from gymnasium.spaces import Box, Discrete

from curiosity.experience import ReplayBuffer
from curiosity.exploration import epsilon_greedy
from curiosity.logging import evaluate

# An implementation of DQN
# Mnih, Volodymyr, et al. Playing Atari with Deep Reinforcement Learning. 2013.

# Experimentation and Logging
SEED = 0
FRAMES_PER_EPOCH = 1000
# Environment
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ENVIRONMENT = 'CartPole-v1'
FRAME_SKIP = 1
GAMMA = 0.99
# Critic
FEATURES = 32
# Replay Buffer
REPLAY_BUFFER_CAPACITY = 100000
# Training
    # Gradient
TOTAL_FRAMES = 1000000
MINIBATCH_SIZE = 128
INITIAL_COLLECTION_SIZE = 10000 # Data to collect before first update
UPDATE_PERIOD = 1 # Data to collect between each update
OPTIMISER = optim.Adam
    # Exploration
INITIAL_EXPLORATION_FACTOR = 1
FINAL_EXPLORATION_FACTOR = 0.1
EXPLORATION_ANNEAL_TIME = 50000

def train():
    random.seed(SEED)

    # Create Environments
    env_train = AutoResetWrapper(gym.make(ENVIRONMENT))
    env_eval = gym.make(ENVIRONMENT)
    if not isinstance(env_train.action_space, Discrete):
        raise ValueError("DQN requires discrete actions")
    if not isinstance(env_train.observation_space, Box) or len(env_train.observation_space.shape) != 1:
        raise ValueError("Critic is defined only for 1D box observations. Is this a pixel space?")
    env_train.reset(seed=SEED)
    env_eval.reset(seed=SEED)
    # Define Critic
    critic = nn.Sequential(
        nn.LazyLinear(out_features=FEATURES),
        nn.Tanh(),
        nn.Linear(in_features=FEATURES, out_features=FEATURES),
        nn.Tanh(),
        nn.Linear(in_features=FEATURES, out_features=env_train.action_space.n),
        nn.LeakyReLU(),
    ).to(device=DEVICE)
    # Initialise Replay Buffer
    memory = ReplayBuffer(
        capacity=REPLAY_BUFFER_CAPACITY,
        shape=(
            env_train.observation_space.shape,
            env_train.action_space.shape,
            1,
            env_train.observation_space.shape
            ),
        dtype=(
            torch.float32,
            torch.int,
            torch.float32,
            torch.float32
            ),
        device=DEVICE
    )
    # Loss
    loss = nn.MSELoss()
    optim = OPTIMISER(params=critic.parameters())

    # Training loop
    obs, _ = env_train.reset()
    obs = torch.tensor(obs, device=DEVICE)
    epoch = 0
    with tqdm(total=TOTAL_FRAMES // FRAMES_PER_EPOCH) as pbar:

        # For logging
        critic_loss = 0

        for step in range(TOTAL_FRAMES // FRAME_SKIP):
            epsilon = min(1,step / EXPLORATION_ANNEAL_TIME) * FINAL_EXPLORATION_FACTOR + (1-min(1,step / EXPLORATION_ANNEAL_TIME)) * INITIAL_EXPLORATION_FACTOR
            # Calculate action
            action = epsilon_greedy(torch.argmax(critic(obs)), env_train.action_space, epsilon)
            # Step environment
            reward = 0
            for _ in range(FRAME_SKIP):
                n_obs, n_reward, n_terminated, _, _ = env_train.step(action)
                reward += n_reward
                if n_terminated: # New episode
                    break
            # Update memory buffer
                # (s_t, a_t, r_t, s_t+1)
            action = torch.tensor(action, device=DEVICE)
            n_obs = torch.tensor(n_obs, device=DEVICE)
            reward = torch.tensor([reward], device=DEVICE)
            transition_tuple = (obs, action, reward, n_obs)

            memory.append(transition_tuple)
            # Keep populating data
            if step < INITIAL_COLLECTION_SIZE:
                continue

            if step % UPDATE_PERIOD == 0:
                # DQN Update
                optim.zero_grad()
                s_t0, a_t, r_t, s_t1 = memory.sample(MINIBATCH_SIZE, continuous=False)
                    # TODO(mark) Consider terminal states
                y = (r_t + torch.max(critic(s_t1)) * GAMMA).squeeze()
                output = loss(critic(s_t0)[torch.arange(MINIBATCH_SIZE), a_t], y)
                critic_loss = output.item()
                output.backward()
                optim.step() 

            # Epoch Logging
            if step % FRAMES_PER_EPOCH == 0:
                pbar.update(1)
                epoch += 1
                reward = evaluate(env=env_eval, policy = lambda x: torch.argmax(critic(torch.tensor(x))).cpu().numpy())
                print(f"epoch {epoch} reward {reward} recent loss {critic_loss} exploration factor {epsilon}")

            # TODO(mark) WandB Logging

if __name__ == "__main__":
    train()