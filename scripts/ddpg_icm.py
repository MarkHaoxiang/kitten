import sys
from typing import Dict

import gymnasium as gym
from gymnasium.wrappers.autoreset import AutoResetWrapper
import torch
from tqdm import tqdm

from curiosity.collector import build_collector
from curiosity.experience import build_replay_buffer
from curiosity.logging import EvaluationEnv, CuriosityArgumentParser
from curiosity.rl import DeepDeterministicPolicyGradient
from curiosity.util import build_actor, build_critic, build_icm, global_seed

parser = CuriosityArgumentParser(
    prog="DDPG+ICM",
    description="Lillicrap, et al. Continuous control with deep reinforcement learning. 2015. Pathak, Deepak, et al. Curiosity-Driven Exploration by Self-Supervised Prediction. 2017. Adds intrinsic curiosity, a type of exploration world model."
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
          features: int = 128,
          target_update_frequency: int = 1,
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
    """ Train DDPG

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
        target_update_frequency (int): Frames between each network update
        replay_buffer_capacity (int): Capacity of replay buffer
        total_frames (int): Total frames to train on
        minibatch_size (int): Batch size to run gradient descent
        initial_collection_size (int): Early start to help with explorations by taking random actions
        tau (float): Speed at which target network follows main networks
        lr (float): Optimiser step size
            # Intrinsic Curiosity
        icm_encoding_features: Size of the latent environment encoding
        eta: Intrinsic reward scale
        beta: Balances between forward and inverse

    Raises:
        ValueError: Invalid configuration passed
    """
    # Metadata
    PROJECT_NAME = parser.generate_project_name(environment, "ddpg_icm")

    # Create Environments
    env = AutoResetWrapper(gym.make(environment, render_mode="rgb_array"))
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
    ddpg = DeepDeterministicPolicyGradient(
        actor=build_actor(env, features, exploration_factor),
        critic=build_critic(env, features),
        gamma=discount_factor,
        lr=lr,
        tau=tau,
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
    collector = build_collector(policy=lambda x: ddpg.actor(x, noise=True), env=env, memory=memory, device=DEVICE)

    # Loss
    optim_icm = torch.optim.Adam(params=icm.parameters(), lr=lr)

    # Logging
    checkpoint = frames_per_checkpoint > 0
    if checkpoint:
        evaluator.register(ddpg.actor.net, "actor")
        evaluator.register(ddpg.critic.net, "critic")

    # Training loop
    epoch = 0
    with tqdm(total=total_frames // frames_per_epoch, file=sys.stdout) as pbar:
        collector.early_start(initial_collection_size)
        for step in range(total_frames):
            collector.collect(n=1)

            # ====================
            # Gradient Update 
            # Mostly Update this
            # ====================

            # DDPG Update
                # Critic
            if step % target_update_frequency == 0:
                s_0, a, r_e, s_1, d = memory.sample(minibatch_size, continuous=False)
                # Add intrinsic reward
                r_i = icm(s_0, s_1, a)
                r_t = r_e + r_i

                loss_critic_value, loss_actor_value = ddpg.update((s_0, a, r_t, s_1, d))
                    # ICM
                loss_icm = icm.calc_loss(s_0, s_1, a)
                optim_icm.zero_grad()
                loss_icm_value = loss_icm.item()
                loss_icm.backward()
                optim_icm.step()

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
                    "train/wall_time": evaluator.get_wall_time(),
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
