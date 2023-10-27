import random
import sys
from typing import Dict

import gymnasium as gym
from gymnasium.wrappers.autoreset import AutoResetWrapper
import torch
from tqdm import tqdm

from curiosity.experience import (
    build_replay_buffer,
    early_start
)
from curiosity.logging import EvaluationEnv, CuriosityArgumentParser
from curiosity.util import (
    build_actor,
    build_critic
)
from curiosity.rl import DeepDeterministicPolicyGradient

parser = CuriosityArgumentParser(
    prog="DDPG",
    description="Lillicrap, et al. Continuous control with deep reinforcement learning. 2015."
)

# An implementation of DDPG
# Lillicrap, et al. Continuous control with deep reinforcement learning. 2015.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(config: Dict = {},
          seed: int = 0,
          frames_per_epoch: int = 1000,
          frames_per_video: int = -1,
          eval_repeat: int = 10,
          wandb: bool = False,
          frames_per_checkpoint: int = 10000,
          environment: str = "Pendulum-v1",
          discount_factor: float = 0.99,
          exploration_factor: float = 0.1,
          features: int = 128,
          target_update_frequency: int = 1,
          replay_buffer_capacity: int = 10000,
          total_frames: int = 50000,
          minibatch_size: int = 128,
          initial_collection_size: int = 1000,
          tau: float = 0.005,
          lr: float = 0.0003,
          **kwargs):
    """ Train DDPG

    Args:
            # Metadata
        config (Dict): Dictionary containing all arguments
            # Experimentation and Logging
        seed (int): Random seed used to setup experiment
        frames_per_epoch (int): Number of frames between each logging point
        frames_per_video (int): Number of frames between each video
        eval_repeat (int): Number of episodes to run in an evaluation
        wandb (bool): Toggle to log to Weights&Biases
            # Environment
        environment (str): Environment name in registry. Defaults to "Pendulum-v1"
        discount_factor (float): Reward discount
            # Algorithm
        exploration_factor (float): Gaussian noise standard deviation for exploration
        features (int): Helps define the complexity of neural nets used
        target_update_frequency (int): Frames between each network update
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
    PROJECT_NAME = parser.generate_project_name(environment, "ddpg")
    random.seed(seed)
    torch.manual_seed(seed)

    # Create Environments
    env = AutoResetWrapper(gym.make(environment, render_mode="rgb_array"))
    env.reset(seed=seed)

    # Define Actor / Critic
    ddpg = DeepDeterministicPolicyGradient(
        actor=build_actor(env, features, exploration_factor),
        critic=build_critic(env, features),
        gamma=discount_factor,
        lr=lr,
        tau=tau,
        device=DEVICE
    )

    # Initialise Replay Buffer    
    memory = build_replay_buffer(env, capacity=replay_buffer_capacity, device=DEVICE)

    # Logging
    evaluator = EvaluationEnv(
        env=env,
        project_name=PROJECT_NAME,
        config=config,
        video=eval_repeat*frames_per_video//frames_per_epoch if frames_per_video > 0 else None,
        wandb_enable=wandb,
        seed=seed,
        device=DEVICE
    )

    checkpoint = frames_per_checkpoint > 0
    if checkpoint:
        evaluator.register(ddpg.actor.net, "actor")
        evaluator.register(ddpg.critic.net, "critic")
        evaluator.checkpoint_registered()
    # Training loop
    obs, _ = env.reset()
    epoch = 0
    with tqdm(total=total_frames // frames_per_epoch, file=sys.stdout) as pbar:
        early_start(env, memory, initial_collection_size)
        # Logging
        for step in range(total_frames):
            # ====================
            # Data Collection
            # ====================

            # Calculate action
            with torch.no_grad():
                action = ddpg.actor(torch.tensor(obs, device=DEVICE, dtype=torch.float32), noise=True) \
                    .cpu().detach().numpy() \
                    .clip(env.action_space.low, env.action_space.high)
            # Step environment
            n_obs, reward, terminated, _, _ = env.step(action)
            # Update memory buffer
                # (s_t, a_t, r_t, s_t+1, d)
            memory.append((obs, action, reward, n_obs, terminated))
            obs = n_obs

            # ====================
            # Gradient Update 
            # Mostly Update this
            # ====================

            # DDPG Update
                # Critic
            if step % target_update_frequency == 0:
                loss_critic_value, loss_actor_value = ddpg.update(memory.sample(minibatch_size, continuous=False))

            # Epoch Logging
            if step % frames_per_epoch == 0:
                epoch += 1
                reward = evaluator.evaluate(
                    policy = lambda x: ddpg.actor(torch.tensor(x, device=DEVICE, dtype=torch.float32)).cpu().numpy(),
                    repeat=eval_repeat
                )
                critic_value =  ddpg.critic(torch.cat((evaluator.saved_reset_states, ddpg.actor(evaluator.saved_reset_states)), 1)).mean().item()
                evaluator.log({
                    "train/frame": step,
                    "train/critic_loss": loss_critic_value,
                    "train/actor_loss": loss_actor_value,
                    "train/critic_value": critic_value 
                })
                pbar.set_description(f"epoch {epoch} reward {reward} critic loss {loss_critic_value} actor loss {loss_actor_value}")
                pbar.update(1)

            if checkpoint and step % frames_per_checkpoint == 0:
                evaluator.checkpoint_registered(step)

        evaluator.close()
        env.close()

if __name__ == "__main__":
    parser.run(train)
