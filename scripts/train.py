import sys

from omegaconf import DictConfig
import hydra
import torch
from tqdm import tqdm

from curiosity.rl import HasCritic
from curiosity.experience import Transition
from curiosity.experience.util import build_collector, build_replay_buffer, build_transition_from_list
from curiosity.policy import ColoredNoisePolicy
from curiosity.util.util import build_env, build_rl, build_intrinsic, global_seed
from curiosity.logging import CuriosityEvaluator, CuriosityLogger, CriticValue

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def train(cfg: DictConfig) -> None:
    # Environment
    env = build_env(**cfg.env)

    # Logging and Evaluation
    logger = CuriosityLogger(cfg, cfg.algorithm.type, path=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    evaluator = CuriosityEvaluator(env, device=DEVICE, **cfg.log.evaluation)
        # TODO Move logging and evaluation down by adding seeding ability to ColoredNoisePolicy

    # RNG
    rng = global_seed(cfg.seed, env, evaluator)

    # Define actor-critic policy
    algorithm = build_rl(env, cfg.algorithm, device=DEVICE)
    policy = ColoredNoisePolicy(algorithm.policy_fn, env.action_space, env.spec.max_episode_steps, rng=rng, device=DEVICE, **cfg.noise)

    # Define intrinsic reward
    intrinsic = build_intrinsic(env, cfg.intrinsic, device=DEVICE)

    # Data pipeline
    memory = build_replay_buffer(env, device=DEVICE, **cfg.memory)
    if cfg.memory.normalise_observation: #TODO: This section can be improved
        normalise_observation = memory.transforms[0]
        policy.fn = normalise_observation.append(policy.fn, bind_method_type=False)
    collector = build_collector(policy, env, memory, device=DEVICE)
    evaluator.policy = policy

    # Register logging and checkpoints
    logger.register_models(algorithm.get_models())
    logger.register_providers([(algorithm, "train"), (intrinsic, "intrinsic"), (collector, "collector"), (evaluator, "evaluation"), (memory, "memory")])
    if isinstance(algorithm, HasCritic): #TODO: This should be an utility
        logger.register_provider(CriticValue(algorithm, evaluator), "train")

    # Training Loop
        # Early start intialisation
    batch = build_transition_from_list(collector.early_start(cfg.train.initial_collection_size), device=DEVICE)
    intrinsic.initialise(batch)
    normalise_observation.add_tensor_batch(batch.s_1) if cfg.memory.normalise_observation else None
        # Main Loop
    pbar = tqdm(total=cfg.train.total_frames // cfg.log.frames_per_epoch, file=sys.stdout)
    for step in range(1, cfg.train.total_frames+1):
        collected = build_transition_from_list(collector.collect(n=1), device=DEVICE)
        normalise_observation.add_tensor_batch(collected.s_1) if cfg.memory.normalise_observation else None
        # RL Update
        batch, aux = memory.sample(cfg.train.minibatch_size)
        batch = Transition(*batch) #TODO: Use an utility wrapper instead
            # Intrinsic Update
        r_t, _, _ = intrinsic.reward(batch)
        intrinsic.update(batch, aux, step=step)
        batch = Transition(batch.s_0, batch.a, r_t, batch.s_1, batch.d)
            # Algorithm Update
        algorithm.update(batch, aux, step=step)
        # Epoch Logging
        if  step % cfg.log.frames_per_epoch == 0:
            pbar.set_description(f"epoch {logger.epoch()} reward {evaluator.evaluate(policy)}"), pbar.update(1)
        # Update checkpoints
        if step % cfg.log.checkpoint.frames_per_checkpoint == 0:
            logger.checkpoint_registered(step)

    # Close resources
    env.close(), evaluator.close(), logger.close(), pbar.close()

if __name__ == "__main__":
    train()