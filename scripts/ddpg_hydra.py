import sys

from omegaconf import DictConfig
import hydra
import torch
from tqdm import tqdm

from curiosity.collector import build_collector
from curiosity.experience import build_replay_buffer
from curiosity.policy import ColoredNoisePolicy
from curiosity.rl.ddpg import DeepDeterministicPolicyGradient
from curiosity.util import build_env, build_actor, build_critic, global_seed
from curiosity.logging import CuriosityEvaluator, CuriosityLogger

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def train(cfg: DictConfig) -> None:

    # Environment
    env = build_env(cfg.env)

    # Logging and Evaluation
    logger = CuriosityLogger(cfg, "ddpg")
    evaluator = CuriosityEvaluator(env, device=DEVICE, **cfg.log.evaluation)
    rng = global_seed(cfg.seed, env, evaluator)

    # Define actor-critic policy
    ddpg = DeepDeterministicPolicyGradient(build_actor(env, **cfg.actor), build_critic(env, **cfg.critic), device=DEVICE, **cfg.ddpg)
    policy = ColoredNoisePolicy(ddpg.actor,env.action_space, env.spec.max_episode_steps, rng=rng, device=DEVICE, **cfg.noise)

    # Data pipeline
    memory = build_replay_buffer(env, device=DEVICE, **cfg.memory)
    collector = build_collector(policy, env, memory, device=DEVICE)
    evaluator.policy = policy

    # Register logging and checkpoints
    logger.register_models([(ddpg.actor.net, "actor"), (ddpg.critic.net, "critic")])
    logger.register_provider([(ddpg, "train"), (evaluator, "evaluation")])

    # Training Loop
    collector.early_start(cfg.train.initial_collection_size)
    with tqdm(total=cfg.train.total_frames // cfg.log.frames_per_epoch, file=sys.stdout) as pbar:
        for step in range(1,  cfg.train.total_frames+1):
            collector.collect(n=1)
            # RL Update
            if step % cfg.ddpg.target_update_frequency == 0:
                ddpg.update(memory.sample(cfg.train.minibatch_size))

            # Epoch Logging
            if step % cfg.log.frames_per_epoch == 0:
                reward = evaluator.evaluate(policy)
                epoch = logger.epoch() 
                logger.log({"train/frame": step})
                pbar.set_description(f"epoch {epoch} reward {reward}")
                pbar.update(1)

            # Update checkpoints
            if step % cfg.log.checkpoint.frames_per_checkpoint == 0:
                logger.checkpoint_registered(step)

        logger.close(), evaluator.close(), env.close()

if __name__ == "__main__":
    train()