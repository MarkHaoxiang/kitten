import sys
from typing import Dict

import gymnasium as gym
from gymnasium.wrappers.autoreset import AutoResetWrapper
import torch
from tqdm import tqdm

from curiosity.collector import build_collector
from curiosity.experience import build_replay_buffer
from curiosity.logging import EvaluationEnv, CuriosityArgumentParser
from curiosity.rl import TwinDelayedDeepDeterministicPolicyGradient
from curiosity.policy import ColoredNoisePolicy
from curiosity.util import build_actor, build_critic, global_seed

ALGORITHM = "td3"
DESCRIPTION = "Fujimoto, et al. Addressing Function Approximation Error in Actor-Critic Methods. 2018."
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = CuriosityArgumentParser(prog=ALGORITHM,description=DESCRIPTION)

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
          exploration_beta: float = 1,
          target_noise: float = 0.1,
          target_noise_clip: float = 0.2,
          features: int = 128,
          critic_update_frequency: int = 1,
          policy_update_frequency: float = 2.0,
          replay_buffer_capacity: int = 10000,
          total_frames: int = 50000,
          minibatch_size: int = 128,
          initial_collection_size: int = 1000,
          tau: float = 0.005,
          lr: float = 0.0003,
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
        exploration_beta (float): Colour of noise for exploration
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

    Raises:
        ValueError: Invalid configuration passed
    """
    
    # Metadata
    PROJECT_NAME = parser.generate_project_name(environment, ALGORITHM)
    policy_update_frequency = int(policy_update_frequency * critic_update_frequency)

    # Create Environments
    env = AutoResetWrapper(gym.make(environment, render_mode="rgb_array"))
    env.reset(seed=seed)
    env_action_scale = torch.tensor(env.action_space.high-env.action_space.low, device=DEVICE) / 2.0
    env_action_min = torch.tensor(env.action_space.low, dtype=torch.float32, device=DEVICE)
    env_action_max = torch.tensor(env.action_space.high, dtype=torch.float32, device=DEVICE)
    evaluator = EvaluationEnv(
        env=env,
        project_name=PROJECT_NAME,
        config=config,
        video=eval_repeat*frames_per_video//frames_per_epoch if frames_per_video > 0 else None,
        wandb_enable=wandb,
        device=DEVICE
    )

    # RNG
    rng = global_seed(seed, env, evaluator)

    # Define Actor / Critic
    td3 = TwinDelayedDeepDeterministicPolicyGradient(
        actor=build_actor(env, features),
        critic_1=build_critic(env, features),
        critic_2=build_critic(env, features),
        gamma=discount_factor,
        lr=lr,
        tau=tau,
        target_noise=target_noise,
        target_noise_clip=target_noise_clip,
        env_action_scale=env_action_scale,
        env_action_min=env_action_min,
        env_action_max=env_action_max,
        device=DEVICE
    )
    policy = ColoredNoisePolicy(
        td3.actor,
        env.action_space,
        episode_length=env.spec.max_episode_steps,
        exploration_factor=exploration_factor,
        beta=exploration_beta,
        rng=rng,
        device=DEVICE
    ) 

    # Initialise Replay Buffer    
    memory = build_replay_buffer(env, capacity=replay_buffer_capacity, device=DEVICE)
    collector = build_collector(policy, env=env, memory=memory, device=DEVICE)

    # Logging
    checkpoint = frames_per_checkpoint > 0
    if checkpoint:
        evaluator.register(td3.actor.net, "actor")
        evaluator.register(td3.critic_1.net, "critic_1")
        evaluator.register(td3.critic_2.net, "critic_2")

    # Training loop
    epoch = 0
    with tqdm(total=total_frames // frames_per_epoch, file=sys.stdout) as pbar:
        collector.early_start(initial_collection_size)
        for step in range(1, total_frames+1):
            collector.collect(n=1)

            # ====================
            # Gradient Update 
            # Mostly Update this
            # ====================
            # TD3 Update
            s_0, a, r, s_1, d = memory.sample(minibatch_size, continuous=False)
            if step % critic_update_frequency == 0:
                # Critic
                loss_critic_value = td3.critic_update((s_0, a, r, s_1, d))

            if step % policy_update_frequency == 0:
                # Actor
                loss_actor_value = td3.actor_update((s_0, a, r, s_1, d))

            # Epoch Logging
            if step % frames_per_epoch == 0 :
                epoch += 1
                reward = evaluator.evaluate(policy, repeat=eval_repeat)
                critic_value = td3.critic_1(torch.cat((evaluator.saved_reset_states, td3.actor(evaluator.saved_reset_states)), 1)).mean().item()
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
