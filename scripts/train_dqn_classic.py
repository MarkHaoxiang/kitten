import random

import torch
import torch.nn as nn
from torch.optim import RMSprop
import gymnasium as gym
from gymnasium.wrappers.autoreset import AutoResetWrapper
from gymnasium.spaces import Box, Discrete

from curiosity.experience import ReplayBuffer
from curiosity.exploration import epsilon_greedy

# An implementation of DQN
# Mnih, Volodymyr, et al. Playing Atari with Deep Reinforcement Learning. 2013.

# Experimentation and Logging
SEED = 0
# Environment
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ENVIRONMENT = 'CartPole-v1'
FRAME_SKIP = 4
GAMMA = 0.99
# Critic
FEATURES = 128
# Replay Buffer
REPLAY_BUFFER_CAPACITY = 1000000
# Training
    # Gradient
#TOTAL_FRAMES = 10000000
TOTAL_FRAMES = 100
MINIBATCH_SIZE = 32
INITIAL_COLLECTION_SIZE = 1000 # Data to collect before first update
UPDATE_PERIOD = 4 # Data to collect between each update
OPTIMISER = RMSprop
    # Exploration
INITIAL_EXPLORATION_FACTOR = 1
FINAL_EXPLORATION_FACTOR = 0.1
EXPLORATION_ANNEAL_TIME = 1000000

def train():
    random.seed(SEED)

    # Create Environments
    env_train = AutoResetWrapper(gym.make(ENVIRONMENT))
    env_eval = AutoResetWrapper(gym.make(ENVIRONMENT))
    if not isinstance(env_train.action_space, Discrete):
        raise ValueError("DQN requires discrete actions")
    if not isinstance(env_train.observation_space, Box) or len(env_train.observation_space.shape) != 1:
        raise ValueError("Critic is defined only for 1D box observations. Is this a pixel space?")
    env_train.reset(seed=SEED)
    env_eval.reset(seed=SEED)
    # Define Critic
    critic = nn.Sequential(
        nn.Linear(in_features=env_train.observation_space.shape[0] + env_train.action_space.n, out_features=FEATURES),
        nn.Tanh(),
        nn.Linear(in_features=FEATURES, out_features=FEATURES),
        nn.Tanh(),
        nn.Linear(in_features=FEATURES, out_features=1),
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
        device=DEVICE
    )
    # Loss
    loss = nn.MSELoss()
    optim = OPTIMISER(params=critic.parameters())

    # Training loop
    obs, _ = env_train.reset()
    obs = torch.tensor(obs, device=DEVICE)
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
            y = r_t + torch.max(critic(s_t1)) * GAMMA
            output = loss(critic(s_t0))
if __name__ == "__main__":
    train()