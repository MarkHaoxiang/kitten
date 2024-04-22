import sys

from omegaconf import DictConfig
import hydra
import torch
from tqdm import tqdm

from kitten.rl.ppo import ProximalPolicyOptimisation
from kitten.rl.advantage import GeneralisedAdvantageEstimator
from kitten.nn import ClassicalDiscreteStochasticActor, ClassicalValue
from kitten.experience.collector import GymCollector
from kitten.experience.util import build_transition_from_list
from kitten.policy import Policy
from kitten.common import global_seed
from kitten.common.util import build_env, build_logger
from kitten.rl.common import td_lambda
from kitten.logging import KittenEvaluator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../config", config_name="ppo")
def train(cfg: DictConfig) -> None:
    # RNG
    rng = global_seed(cfg.seed)

    # Environment
    env = build_env(**cfg.env, seed=rng)

    # Define Value, Actor
    actor = ClassicalDiscreteStochasticActor(env, rng).to(DEVICE)
    value = ClassicalValue(env).to(DEVICE)
    gamma, lmbda = cfg.algorithm.gamma, cfg.algorithm.lmbda
    optim_value = torch.optim.Adam(value.parameters())
    gae = GeneralisedAdvantageEstimator(value, lmbda, gamma)
    ppo = ProximalPolicyOptimisation(actor, gae, rng)
    policy = Policy(fn=ppo.policy_fn, device=DEVICE)

    # Data Pipeline
    collector = GymCollector(env=env, policy=policy, device=DEVICE)

    # Logging and evaluation
    logger = build_logger(cfg)
    evaluator = KittenEvaluator(env, policy=policy, device=DEVICE, **cfg.log.evaluation)

    # Register logging
    logger.register_providers(
        [
            (ppo, "train"),
            (collector, "collector"),
            (evaluator, "evaluation")
        ]
    )
    
    step = 0
    previous_epoch_step = 0
    pbar = tqdm(total=cfg.train.total_frames, file=sys.stdout)
    while step < cfg.train.total_frames:
        # Collect a single episode
        batch = build_transition_from_list(
            collector.collect(
                (
                    env.spec.max_episode_steps
                    if env.spec.max_episode_steps is not None
                    else 500
                ),
                single_episode=True,
            ),
            device=DEVICE,
        )
        step += batch.shape[0]

        # Update Value Function
        value_targets = td_lambda(batch, lmbda, gamma, value)
        total_value_loss = 0
        for _ in range(cfg.algorithm.update_epochs):
            optim_value.zero_grad()
            value_loss = ((value_targets - value.v(batch.s_0)) ** 2).mean()
            total_value_loss += value_loss.item()
            value_loss.backward()
            optim_value.step()

        # Update PPO
        l = ppo.update(batch, aux=None, step=step)
        # Epoch Logging
        if step > previous_epoch_step + 100:
            evaluation_reward = evaluator.evaluate(policy)
            pbar.set_description(
                f"Epoch {logger.epoch()} Evaluation Reward {evaluation_reward}"
            )
            pbar.update(collector.frame)
            previous_epoch_step = step

    # Close resources
    env.close(), evaluator.close(), logger.close(), pbar.close()

if __name__ == "__main__":
    train()
