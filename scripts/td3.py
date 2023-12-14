import sys

from omegaconf import DictConfig
import hydra
import torch
from tqdm import tqdm

from curiosity.collector import build_collector
from curiosity.experience import build_replay_buffer
from curiosity.policy import ColoredNoisePolicy
from curiosity.rl.td3 import TwinDelayedDeepDeterministicPolicyGradient
from curiosity.util import build_env, build_actor, build_critic, global_seed
from curiosity.logging import CuriosityEvaluator, CuriosityLogger

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def train(cfg: DictConfig) -> None:
    # Environment
    env = build_env(cfg.env)
    env_action_scale = torch.tensor(env.action_space.high-env.action_space.low, device=DEVICE) / 2.0
    env_action_min = torch.tensor(env.action_space.low, dtype=torch.float32, device=DEVICE)
    env_action_max = torch.tensor(env.action_space.high, dtype=torch.float32, device=DEVICE)

    # Logging and Evaluation
    logger, evaluator = CuriosityLogger(cfg, "td3"), CuriosityEvaluator(env, device=DEVICE, **cfg.log.evaluation)

    # RNG
    rng = global_seed(cfg.seed, env, evaluator)

    # Define actor-critic policy
    td3 = TwinDelayedDeepDeterministicPolicyGradient(
        build_actor(env, **cfg.actor),
        build_critic(env, **cfg.critic),
        build_critic(env, **cfg.critic),
        device=DEVICE,
        env_action_scale=env_action_scale,
        env_action_min=env_action_min,
        env_action_max=env_action_max,
        **cfg.td3
    )
    policy = ColoredNoisePolicy(td3.actor,env.action_space, env.spec.max_episode_steps, rng=rng, device=DEVICE, **cfg.noise)

    # Data pipeline
    memory = build_replay_buffer(env, device=DEVICE, error_fn=lambda x: torch.abs(td3.td_error(*x)).detach().cpu().numpy(), **cfg.memory)
    collector = build_collector(policy, env, memory, device=DEVICE)
    evaluator.policy = policy

    # Register logging and checkpoints
    logger.register_models([(td3.actor.net, "actor"), (td3.critic_1.net, "critic_1"), (td3.critic_2.net, "critic_2")])
    logger.register_providers([(td3, "train"), (collector, "collector"), (evaluator, "evaluation"), (memory, "memory")])

    # Training Loop
    pbar = tqdm(total=cfg.train.total_frames // cfg.log.frames_per_epoch, file=sys.stdout)
    collector.early_start(cfg.train.initial_collection_size)
    for step in range(1,  cfg.train.total_frames+1):
        collector.collect(n=1)

        # RL Update
        batch, weights = memory.sample(cfg.train.minibatch_size)
        if step % cfg.td3.critic_update_frequency == 0:
            td3.critic_update(batch, weights)
        if step % cfg.td3.policy_update_frequency == 0:
            td3.actor_update(batch, weights)

        # Epoch Logging
        if  step % cfg.log.frames_per_epoch == 0:
            critic_value =  td3.critic(torch.cat((evaluator.saved_reset_states, td3.actor(evaluator.saved_reset_states)), 1)).mean().item()
            logger.log({"train/critic_value": critic_value})
            reward, epoch = evaluator.evaluate(policy), logger.epoch()
            pbar.set_description(f"epoch {epoch} reward {reward}"), pbar.update(1)
        # Update checkpoints
        if step % cfg.log.checkpoint.frames_per_checkpoint == 0:
            logger.checkpoint_registered(step)

    # Close resources
    env.close(), evaluator.close(), logger.close(), pbar.close()

if __name__ == "__main__":
    train()