import sys

from omegaconf import DictConfig
import hydra
import torch

from kitten.rl.ppo import ProximalPolicyOptimisation
from kitten.rl.advantage import GeneralisedAdvantageEstimator
from kitten.nn import (
    ClassicalDiscreteStochasticActor,
    ClassicalDiscreteCritic,
    CriticPolicyPair,
)
from kitten.experience.collector import GymCollector
from kitten.experience.util import build_transition_from_list
from kitten.policy import Policy
from kitten.common import global_seed
from kitten.common.util import build_env

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../config", config_name="ppo")
def train(cfg: DictConfig) -> None:
    # RNG
    rng = global_seed(cfg.seed)

    # Environment
    env = build_env(**cfg.env, seed=rng)
    actor = ClassicalDiscreteStochasticActor(env, rng).to(DEVICE)
    critic = ClassicalDiscreteCritic(env).to(DEVICE)
    value = CriticPolicyPair(critic, lambda s: actor.a(s))
    gae = GeneralisedAdvantageEstimator(value)
    ppo = ProximalPolicyOptimisation(actor, gae, rng)
    collector = GymCollector(
        env=env, policy=Policy(fn=ppo.policy_fn, device=DEVICE), device=DEVICE
    )

    optim_critic = torch.optim.Adam(params=critic.parameters())

    step = 0
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
        # TODO
        ppo.update(batch, aux=None, step=step)
        break


if __name__ == "__main__":
    train()
