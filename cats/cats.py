# Std
import copy
from typing import Any

# Training
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import DictConfig

# Kitten
from kitten.experience import Transitions
from kitten.experience.util import (
    build_replay_buffer,
    build_transition_from_update,
    build_transition_from_list,
)
from kitten.experience.collector import GymCollector
from kitten.policy import ColoredNoisePolicy, Policy
from kitten.common import *
from kitten.common.typing import Device
from kitten.common.util import *

# Cats
from .rl import QTOptCats, ResetValueOverloadAux
from .env import *
from .reset import *
from .teleport import *


class CatsExperiment:
    """Experiment baseline"""

    def __init__(
        self,
        cfg: DictConfig,
        collection_steps: int,
        max_episode_steps: int | None = None,
        death_is_not_the_end: bool = True,
        fixed_reset: bool = True,
        enable_policy_sampling: bool = True,
        reset_as_an_action: float | None = None,
        teleport_strategy: tuple[str, Any] | None = None,
        environment_action_noise: float = 0,
        seed: int = 0,
        deprecated_testing_flag: bool = False,
        device: Device = "cpu",
    ):
        # Parameters
        self.cfg = copy.deepcopy(cfg)
        self.cfg.total_frames = collection_steps
        self.cfg.seed = seed
        self.cfg.env.max_episode_steps = max_episode_steps
        self.device = device

        self.fixed_reset = fixed_reset
        self.enable_policy_sampling = enable_policy_sampling
        self.death_is_not_the_end = death_is_not_the_end
        self.environment_action_noise = environment_action_noise
        self.reset_as_an_action = reset_as_an_action
        self.deprecated_testing_flag = deprecated_testing_flag  # Utility to compare different versions of algorithms for bug testing
        if teleport_strategy is None:
            self.teleport_strategy = None
        else:
            self.teleport_strategy = teleport_strategy[0]
            self.teleport_strategy_parameters = teleport_strategy[1]

        # Init
        self._build_env()
        self._build_policy()
        self._build_data()
        self._build_intrinsic()
        self._teleport_initialisation()

        self.reset_distribution = np.array(
            [self.reset_env() for _ in range(1 if self.fixed_reset else 1024)]
        )
        self.reset_distribution = torch.tensor(
            self.reset_distribution, device=self.device
        )

        # Logging
        self.log = {
            "total_intrinsic_reward": 0,
            "newly_collected_intrinsic_reward": 0,
            "total_collected_extrinsic_reward": 0,
            "truncation_steps": [],
        }

        teleport_rng = self.rng.build_generator()
        if self.teleport_strategy is not None:
            match self.teleport_strategy:
                case "e_greedy":
                    self.teleport_strategy = EpsilonGreedyTeleport(
                        self.algorithm, teleport_rng, self.teleport_strategy_parameters
                    )
                case "thompson":
                    self.teleport_strategy = ThompsonTeleport(
                        self.algorithm, teleport_rng, self.teleport_strategy_parameters
                    )
                case "ucb":
                    self.teleport_strategy = UCBTeleport(
                        self.algorithm, self.teleport_strategy_parameters
                    )
                case _:
                    raise ValueError(
                        f"Unknown Teleport Strategy {self.teleport_strategy}"
                    )
            self.log["teleport_targets_index"] = []
            self.log["teleport_targets_observations"] = []

    def _teleport_initialisation(self):
        self.teleport_memory = LatestEpisodeTeleportMemory(
            self.rng.build_generator(), device=self.device
        )
        self.normalise_observation.prepend(self.teleport_memory.targets)

    def _build_intrinsic(self):
        self.intrinsic = build_intrinsic(
            self.env, self.cfg.intrinsic, device=self.device
        )

    def _build_env(self):
        self.env = build_env(**self.cfg.env)

        # Fixed-Reset
        if self.fixed_reset:
            self.env = FixedResetWrapper(self.env)

        # Random noise to action (aleatoric uncertainty)
        if self.environment_action_noise > 0:
            self.env = StochasticActionWrapper(
                self.env, noise=self.environment_action_noise
            )
        # Reset as an action
        if self.reset_as_an_action is not None:
            self.env = ResetActionWrapper(
                self.env, penalty=self.reset_as_an_action, deterministic=True
            )
        self.rng = global_seed(self.cfg.seed, self.env)

    def _build_policy(self):
        self.algorithm = QTOptCats(
            build_critic=lambda: build_critic(self.env, **self.cfg.algorithm.critic).to(
                device=self.device
            ),
            action_space=self.env.action_space,
            obs_space=self.env.observation_space,
            device=self.device,
            **self.cfg.algorithm,
        )
        self.policy = ColoredNoisePolicy(
            self.algorithm.policy_fn,
            self.env.action_space,
            (
                int(self.env.spec.max_episode_steps)
                if isinstance(self.env.spec.max_episode_steps, int)
                else 1000
            ),
            rng=self.rng.build_generator(),
            device=self.device,
            **self.cfg.noise,
        )

    def _build_data(self):
        self.memory, self.normalise_observation = build_replay_buffer(
            self.env,
            capacity=self.cfg.total_frames,
            normalise_observation=True,
            device=self.device,
        )
        # Remove automatic memory addition for more control
        self.collector = GymCollector(self.policy, self.env, device=self.device)
        self.policy.fn = self.normalise_observation.append(
            self.policy.fn, bind_method_type=False
        )

    def _update_memory(self, obs, action, reward, n_obs, terminated, truncated):
        self.memory.append((obs, action, reward, n_obs, terminated, truncated))

    def V(self, s) -> torch.Tensor:
        """Calculates the value function for states s"""
        return self.algorithm.value.v(s)

    def reset_env(self):
        """Manual reset of the environment"""
        if self.enable_policy_sampling:
            self.algorithm.reset_critic()
        o, _ = self.collector.env.reset()
        self.collector.obs = o
        return o

    def early_start(self, n: int):
        """Overrides early start for tighter control

        Args:
            n (int): number of steps
        """
        self.reset_env()
        policy = self.collector.policy
        self.collector.set_policy(Policy(lambda _: self.env.action_space.sample()))
        results = []
        for i in range(n):
            obs, action, reward, n_obs, terminated, truncated = self.collector.collect(
                n=1, early_start=True
            )[-1]
            if terminated or truncated:
                self.reset_env()
            self._update_memory(obs, action, reward, n_obs, terminated, truncated)
            results.append((obs, action, reward, n_obs, terminated, truncated))
        self.collector.policy = policy
        batch = build_transition_from_list(results, device=self.device)
        self.intrinsic.initialise(batch)
        self.normalise_observation.add_tensor_batch(batch.s_1)

    # ==============================
    # Key Algorithm: Teleportation
    # ==============================
    def _reset(self):
        if isinstance(self.teleport_strategy, TeleportStrategy):
            s = self.teleport_memory.targets()
            tid = self.teleport_strategy.select(s)
            self.log["teleport_targets_index"].append(
                (self.teleport_memory.episode_step, tid)
            )
            self.teleport_memory.select(tid, self.collector)
            self.log["teleport_targets_observations"].append(self.collector.obs)
        else:
            self.reset_env()

    # ==============================

    def run(self):
        """Default Experiment run"""
        # Initialisation
        self.early_start(self.cfg.train.initial_collection_size)
        self.reset_env()
        self.teleport_memory.state = self.env

        # Main Loop
        for step in tqdm(range(1, self.cfg.train.total_frames + 1)):
            # Collect Data
            obs, action, reward, n_obs, terminated, truncated = self.collector.collect(
                n=1
            )[-1]
            self._update_memory(obs, action, reward, n_obs, terminated, truncated)
            self.normalise_observation.add_tensor_batch(
                torch.tensor(n_obs, device=self.device).unsqueeze(0)
            )

            # Teleport
            self.teleport_memory.update(env=self.collector.env, obs=obs)

            if terminated or truncated:
                if truncated:
                    self.log["truncation_steps"].append(
                        self.teleport_memory.episode_step
                    )
                self._reset()

            # Updates
            batch, aux = self.memory.sample(self.cfg.train.minibatch_size)

            # Death is not the end - disabled termination
            if self.death_is_not_the_end:
                batch.d = torch.zeros_like(batch.d, device=self.device).bool()
            # Intrinsic Reward Calculation
            r_t, r_e, r_i = self.intrinsic.reward(batch)
            self.intrinsic.update(batch, aux, step=step)

            # RL Update
            aux = ResetValueOverloadAux(
                weights=aux.weights,
                random=aux.random,
                indices=aux.indices,
                v_1=torch.ones_like(batch.r, device=self.device),
                reset_value_mixin_select=torch.zeros_like(
                    batch.t, device=self.device
                ).bool(),
                reset_value_mixin_enable=False,
            )
            # Update batch
            s_1 = batch.s_1
            # Question: Should policy learn the value of resetting?
            if self.reset_as_an_action is not None:
                reset_sample = self.reset_distribution.to(torch.float32)
                reset_sample = self.normalise_observation.transform(reset_sample)
                critics = random.sample(self.algorithm.critics, 2)
                a_1 = self.algorithm.policy_fn(reset_sample, critic=critics[0].target)
                a_2 = self.algorithm.policy_fn(reset_sample, critic=critics[1].target)
                target_max_1 = critics[0].target.q(reset_sample, a_1).squeeze()
                target_max_2 = critics[1].target.q(reset_sample, a_2).squeeze()
                reset_value = torch.minimum(target_max_1, target_max_2).mean().item()
                aux.v_1 = aux.v_1 * reset_value
                select = batch.t
                if self.death_is_not_the_end:
                    select = torch.logical_or(batch.t, batch.d)
                aux.reset_value_mixin_select = select
                aux.reset_value_mixin_enable = True

                # Since we don't consider extrinsic rewards (for now)
                # Manually add the penalty to intrinsic rewards
                r_i = r_i - batch.t * self.reset_as_an_action

            batch.s_1 = s_1
            batch.r = r_i
            self.algorithm.update(batch, aux, step=step)

            # Log
            self.log["total_intrinsic_reward"] += r_i.mean().item()
            self.log["total_collected_extrinsic_reward"] += reward
            self.log["newly_collected_intrinsic_reward"] += self.intrinsic.reward(
                build_transition_from_update(
                    obs, action, reward, n_obs, terminated, device=self.device
                )
            )[2].item()
