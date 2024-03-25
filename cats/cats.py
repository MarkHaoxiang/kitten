# Std
import copy
from typing import Any, Optional, Tuple

# Training
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import DictConfig
import gymnasium as gym

# Kitten
from kitten.experience.util import (
    Transitions,
    build_transition_from_update,
    build_transition_from_list,
)
from kitten.experience.memory import ReplayBuffer
from kitten.experience.collector import GymCollector
from kitten.policy import ColoredNoisePolicy, Policy
from kitten.common import *
from kitten.common.util import *
from kitten.dataflow.normalisation import RunningMeanVariance

# Cats
from .rl import QTOptCats
from .env import *


class CatsExperiment:
    """Experiment baseline"""

    def __init__(
        self,
        cfg: DictConfig,
        collection_steps: int,
        max_episode_steps: Optional[int] = None,
        death_is_not_the_end: bool = True,
        fixed_reset: bool = True,
        enable_policy_sampling: bool = True,
        reset_as_an_action: Optional[int] = None,
        teleport_strategy: Optional[Tuple[str, Any]] = None,
        environment_action_noise: float = 0,
        enable_reset_sampling: bool = True,
        seed: int = 0,
        device: str = "cpu",
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
        self.enable_reset_sampling = enable_reset_sampling
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
        self.np_rng = np.random.default_rng(self.cfg.seed)

        if self.enable_reset_sampling:
            self.reset_distribution = [self.reset_env() for _ in range(1024)]
            self.reset_distribution = torch.tensor(
                np.array(self.reset_distribution), device=self.device
            )

        # Logging
        self.log = {
            "total_intrinsic_reward": 0,
            "newly_collected_intrinsic_reward": 0,
            "total_collected_extrinsic_reward": 0,
        }
        if not self.teleport_strategy is None:
            self.log["teleport_targets_index"] = []
            self.log["teleport_targets_observations"] = []

    def _teleport_initialisation(self):
        self.teleport_targets_observations = []
        self.teleport_targets_saves = []
        self.current_index = 0
        self.teleport_index = 0

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
        if not self.reset_as_an_action is None:
            self.env = ResetActionWrapper(
                self.env, check_frequency=self.reset_as_an_action
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
        if isinstance(self.env.spec.max_episode_steps, int):
            cycle = self.env.spec.max_episode_steps
        else:
            cycle = 1000
        self.policy = ColoredNoisePolicy(
            self.algorithm.policy_fn,
            self.env.action_space,
            cycle,
            rng=self.rng,
            device=self.device,
            **self.cfg.noise,
        )

    def _build_data(self):
        rmv = RunningMeanVariance()
        self.normalise_observation = rmv
        normalise_observation = [rmv, None, None, rmv, None, None]
        self.memory = ReplayBuffer(
            capacity=self.cfg.total_frames,
            shape=(
                self.env.observation_space.shape,
                self.env.action_space.shape,
                (),
                self.env.observation_space.shape,
                (),
                (),  # Adds Truncation
            ),
            dtype=(
                torch.float32,
                torch.torch.float32,
                torch.float32,
                torch.float32,
                torch.bool,
                torch.bool,
            ),
            transforms=normalise_observation,
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
        target_action = self.algorithm.policy_fn(s)
        V = self.algorithm.critic.target.q(s, target_action)
        return V

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
        self.collector.set_policy(
            Policy(lambda _: self.env.action_space.sample(), transform_obs=False)
        )
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
    def _teleport_update(self, obs):
        self.state = copy.deepcopy(self.collector.env)
        self.teleport_targets_observations.append(obs)
        self.teleport_targets_saves.append(self.state)
        self.current_index += 1

    def _teleport_selection(self, V: torch.Tensor):
        """Given the values of teleport targets, select a goal"""
        if self.teleport_strategy == "e_greedy":
            teleport_index = torch.argmax(V).item()
            if self.np_rng.random() <= self.teleport_strategy_parameters:
                teleport_index = 0
        elif self.teleport_strategy == "thompson":
            V = V**self.teleport_strategy_parameters
            p = V / V.sum()
            pt = self.np_rng.random()
            pc = 0
            for i, pi in enumerate(p):
                pc += pi
                if pc >= pt or i == len(p) - 1:
                    teleport_index = i
                    break
        else:
            raise ValueError(f"Unknown teleport strategy {self.teleport_strategy}")

        return teleport_index

    def _reset(self):
        if self.teleport_strategy is None:
            self.reset_env()
        else:
            # Call Teleportation
            V = self.V(
                torch.tensor(
                    self.teleport_targets_observations[: self.current_index],
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            with torch.no_grad():
                self.teleport_index = self._teleport_selection(V)
            # TODO: Selecting Resets

            self.current_index = self.teleport_index
            self.collector.env = self.teleport_targets_saves[self.teleport_index]
            self.collector.obs = self.teleport_targets_observations[self.teleport_index]
            self.collector.env.np_random = np.random.default_rng(
                self.np_rng.integers(65536)
            )
            self.teleport_targets_observations = self.teleport_targets_observations[
                : self.teleport_index + 1
            ]
            self.teleport_targets_saves = self.teleport_targets_saves[
                : self.teleport_index + 1
            ]
            self.state = copy.deepcopy(self.collector.env)

            # Logging
            self.log["teleport_targets_index"].append(self.teleport_index)
            self.log["teleport_targets_observations"].append(self.collector.obs)

    # ==============================

    def run(self):
        """Default Experiment run"""
        # Initialisation
        self.early_start(self.cfg.train.initial_collection_size)
        self.reset_env()

        # Main Loop
        for step in tqdm(range(1, self.cfg.train.total_frames + 1)):
            # Collect Data
            obs, action, reward, n_obs, terminated, truncated = self.collector.collect(
                n=1
            )[-1]
            # TODO: If resets are controllable, n_obs should be stored in RB for Policy, but not intrinsic
            # Or should it be stored for intrinsic?
            self._update_memory(obs, action, reward, n_obs, terminated, truncated)
            self.normalise_observation.add_tensor_batch(
                torch.tensor(n_obs, device=self.device).unsqueeze(0)
            )

            # Teleport
            if not self.teleport_strategy is None:
                self._teleport_update(obs)
            if truncated:
                self.current_index = 0

            if terminated or truncated and self.current_index > 0:
                self._reset()

            # Updates
            data, aux = self.memory.sample(self.cfg.train.minibatch_size)
            batch = Transitions(*data[:5])
            batch_death = batch.d
            batch_truncated = data[-1]

            # Death is not the end - disabled termination
            if self.death_is_not_the_end:
                batch = Transitions(
                    batch.s_0,
                    batch.a,
                    batch.r,
                    batch.s_1,
                    torch.zeros(batch.d.shape, device=self.device).bool(),
                )

            # Intrinsic Reward Calculation
            r_t, r_e, r_i = self.intrinsic.reward(batch)
            self.intrinsic.update(batch, aux, step=step)

            # RL Update
            # Update batch
            s_1 = batch.s_1
            # Question: Should policy learn the value of resetting?
            if self.enable_reset_sampling:
                indices = torch.randint(
                    0, len(self.reset_distribution), (len(s_1),), device=self.device
                )
                reset_sample = self.reset_distribution[indices]
                if self.reset_as_an_action:
                    s_1 = torch.where(
                        input=reset_sample,
                        other=s_1,
                        condition=batch_truncated.unsqueeze(-1),
                    )
                if self.death_is_not_the_end:
                    s_1 = torch.where(
                        input=reset_sample,
                        other=s_1,
                        condition=batch_death.unsqueeze(-1),
                    )

            batch = Transitions(batch.s_0, batch.a, r_i, s_1, batch.d)
            self.algorithm.update(batch, aux, step=step)

            # Log
            self.log["total_intrinsic_reward"] += r_i.mean().item()
            self.log["total_collected_extrinsic_reward"] += reward
            self.log["newly_collected_intrinsic_reward"] += self.intrinsic.reward(
                build_transition_from_update(
                    obs, action, reward, n_obs, terminated, device=self.device
                )
            )[2].item()
