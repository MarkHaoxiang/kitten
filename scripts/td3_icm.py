import sys
from typing import Dict

import gymnasium as gym
from gymnasium.wrappers.autoreset import AutoResetWrapper
import torch
from tqdm import tqdm

from curiosity.experience import build_replay_buffer, early_start
from curiosity.logging import EvaluationEnv, CuriosityArgumentParser
from curiosity.rl import TwinDelayedDeepDeterministicPolicyGradient
from curiosity.util import build_actor, build_critic, build_icm, global_seed

parser = CuriosityArgumentParser(
    prog="TD3+ICM",
    description= """Fujimoto, et al. Addressing Function Approximation Error in Actor-Critic Methods. 2018. Adds clipped double critic, delayed policy updates, and value function smoothing  to DDPG. Pathak, Deepak, et al. Curiosity-Driven Exploration by Self-Supervised Prediction. 2017. Adds intrinsic curiosity, a type of exploration world model."""
)

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
          icm_encoding_features: int = 32,
          eta: float = 1,
          beta: float = 0.2,
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
    PROJECT_NAME = parser.generate_project_name(environment, "td3_icm")
    policy_update_frequency = int(policy_update_frequency * critic_update_frequency)

    # Create Environments
    env = AutoResetWrapper(gym.make(environment, render_mode="rgb_array"))
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
    global_seed(seed, env, evaluator)

    # Define Actor / Critic
    td3 = TwinDelayedDeepDeterministicPolicyGradient(
        actor=build_actor(env, features, exploration_factor),
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

    # Define curiosity
    icm = build_icm(
        env=env,
        encoding_size=icm_encoding_features,
        eta=eta,
        beta=beta,
        device=DEVICE
    )

    # Initialise Replay Buffer    
    memory = build_replay_buffer(env, capacity=replay_buffer_capacity, device=DEVICE)

    # Loss
    optim_icm = torch.optim.Adam(params=icm.parameters(), lr=lr)

    # Logging
    checkpoint = frames_per_checkpoint > 0
    if checkpoint:
        evaluator.register(td3.actor.net, "actor")
        evaluator.register(td3.critic_1.net, "critic_1")
        evaluator.register(td3.critic_2.net, "critic_2")
        evaluator.checkpoint_registered()

    # Training loop
    obs, _ = env.reset(seed=seed)
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
                action = td3.actor(torch.tensor(obs, device=DEVICE, dtype=torch.float32), noise=True) \
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
            s_0, a, r_e, s_1, d = memory.sample(minibatch_size, continuous=False)
                # Add intrinsic reward
            r_i = icm(s_0, s_1, a)
            r_t = r_e + r_i

            if step % critic_update_frequency == 0:
                # Critic
                loss_critic_value = td3.critic_update((s_0, a, r_t, s_1, d))

            if step % policy_update_frequency == 0:
                # Actor
                loss_actor_value = td3.actor_update((s_0, a, r_t, s_1, d))

                # Update icm
                loss_icm = icm.calc_loss(s_0, s_1, a)
                optim_icm.zero_grad()
                loss_icm_value = loss_icm.item()
                loss_icm.backward()
                optim_icm.step()

            # Epoch Logging
            if step % frames_per_epoch == 0 :
                epoch += 1
                reward = evaluator.evaluate(
                    policy = lambda x: td3.actor(torch.tensor(x, device=DEVICE, dtype=torch.float32)).cpu().numpy(),
                    repeat=eval_repeat
                )
                critic_value = td3.critic_1(torch.cat((evaluator.saved_reset_states, td3.actor(evaluator.saved_reset_states)), 1)).mean().item()
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
                evaluator.checkpoint_registered(step)

        evaluator.close()
        env.close()

if __name__ == "__main__":
    parser.run(train)