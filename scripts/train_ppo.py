from omegaconf import DictConfig
import hydra
import torch

from kitten.rl.ppo import ProximalPolicyOptimisation
from kitten.rl.advantage import GeneralisedAdvantageEstimator
from kitten.nn import (
    ClassicalDiscreteStochasticActor,
    ClassicalValue
)
from kitten.experience.collector import GymCollector
from kitten.experience.util import build_transition_from_list
from kitten.policy import Policy
from kitten.common import global_seed
from kitten.common.util import build_env
from kitten.rl.common import td_lambda

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base=None, config_path="../config", config_name="ppo")
def train(cfg: DictConfig) -> None:
    # RNG
    rng = global_seed(cfg.seed)

    # Environment
    env = build_env(**cfg.env, seed=rng)
    actor = ClassicalDiscreteStochasticActor(env, rng).to(DEVICE)
    value = ClassicalValue(env).to(DEVICE)
    gamma, lmbda = cfg.algorithm.gamma, cfg.algorithm.lmbda
    optim_value = torch.optim.Adam(value.parameters())
    gae = GeneralisedAdvantageEstimator(value, lmbda, gamma)
    ppo = ProximalPolicyOptimisation(actor, gae, rng)
    collector = GymCollector(
        env=env, policy=Policy(fn=ppo.policy_fn, device=DEVICE), device=DEVICE
    )

    step = 0

    previous_epoch_step = 0
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
            value_loss = ((value_targets - value.v(batch.s_0))**2).mean()
            total_value_loss += value_loss.item()
            value_loss.backward()
            optim_value.step()

        # Update PPO
        l = ppo.update(batch, aux=None, step=step)

        if step > previous_epoch_step + 100:
            print(f"{step} | Actor Loss {l} Value Loss {total_value_loss} Ep Length {len(batch)}")
            previous_epoch_step = step


if __name__ == "__main__":
    train()
