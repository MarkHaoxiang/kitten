# Std
import copy


# Training
import torch
from tqdm import tqdm
from omegaconf import DictConfig

# Kitten
from kitten.experience.util import (
    build_replay_buffer,
    build_transition_from_list,
)
from kitten.experience.collector import GymCollector
from kitten.policy import ColoredNoisePolicy, Policy
from kitten.common import *
from kitten.common.typing import Device
from kitten.common.util import *
from kitten.logging import DictEngine, KittenLogger

# Cats
from .rl import QTOptCats, ResetValueOverloadAux
from .env import *
from .reset import *
from .teleport import *
from .logging import *


class CatsExperiment:
    """Experiment baseline"""

    def __init__(
        self,
        cfg: DictConfig,
        deprecated_testing_flag: bool = False,
        device: Device = "cpu",
    ):
        # Parameters
        self.cfg = copy.deepcopy(cfg)
        self.device = device

        self.fixed_reset = self.cfg.cats.fixed_reset
        self.enable_policy_sampling = self.cfg.cats.enable_policy_sampling
        self.death_is_not_the_end = self.cfg.cats.death_not_end
        self.environment_action_noise = self.cfg.cats.env_action_noise
        self.reset_as_an_action = self.cfg.cats.reset_action
        self.deprecated_testing_flag = deprecated_testing_flag  # Utility to compare different versions of algorithms for bug testing
        self.logging_path = self.cfg.log.path

        # Init
        self._build_env()
        self._build_policy()
        self._build_data()
        self._build_intrinsic()
        self._build_teleport()
        self._build_logging()

    def _build_teleport(self):
        teleport_rng = self.rng.build_generator()
        self.teleport_cfg = self.cfg.cats.teleport
        if self.teleport_cfg.enable:
            match self.teleport_cfg.type:
                case "e_greedy":
                    self.teleport_strategy = EpsilonGreedyTeleport(
                        algroithm=self.algorithm,
                        rng=teleport_rng,
                        **self.teleport_cfg.kwargs,
                    )
                case "thompson":
                    self.teleport_strategy = ThompsonTeleport(
                        algorithm=self.algorithm,
                        rng=teleport_rng,
                        **self.teleport_cfg.kwargs,
                    )
                case "ucb":
                    self.teleport_strategy = UCBTeleport(
                        algorithm=self.algorithm, **self.teleport_cfg.kwargs
                    )
                case _:
                    raise ValueError(
                        f"Unknown Teleport Strategy {self.teleport_cfg.type}"
                    )

        match self.teleport_cfg.memory.type:
            case "fifo":
                self.teleport_memory = FIFOTeleportMemory(
                    env=self.env,
                    rng=self.rng.build_generator(),
                    capacity=self.teleport_cfg.memory.capacity,
                    device=self.device,
                )
            case "episode":
                self.teleport_memory = LatestEpisodeTeleportMemory(
                    rng=self.rng.build_generator(), device=self.device
                )
            case _:
                raise ValueError(f"Unknown Teleport Memory")

        self.reset_memory = ResetBuffer(
            env=self.env,
            capacity=1 if self.fixed_reset else 128,
            rng=self.rng.build_generator(),
            device=self.device,
        )

        self.rmv.prepend(self.reset_memory.targets)
        self.rmv.prepend(self.teleport_memory.targets)

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
        if self.reset_as_an_action.enable:
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
            rng=self.rng.build_generator(),
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
        if self.reset_as_an_action.enable:
            self.policy = ResetPolicy(env=self.env, policy=self.policy)

    def _build_data(self):
        self.memory, self.rmv = build_replay_buffer(
            self.env,
            capacity=self.cfg.train.total_frames
            + self.cfg.train.initial_collection_size,
            normalise_observation=True,
            device=self.device,
        )
        # Remove automatic memory addition for more control
        self.collector = GymCollector(self.policy, self.env, device=self.device)
        self.rmv.append(self.policy.fn)

    def _update_memory(self, obs, action, reward, n_obs, terminated, truncated):
        self.memory.append((obs, action, reward, n_obs, terminated, truncated))

    def _build_logging(self):
        self.logger = KittenLogger(
            self.cfg, algorithm="cats", engine=DictEngine, path=self.logging_path
        )
        self.logger.register_providers(
            [
                (self.algorithm, "train"),
                (self.intrinsic, "intrinsic"),
                (self.collector, "collector"),
                (self.memory, "memory"),
            ]
        )

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
        if not self.reset_as_an_action.enable:
            self.collector.set_policy(Policy(lambda _: self.env.action_space.sample()))
        else:
            # TODO: Edit reset evaluation policy with a small probability of of resets,
            # But teleport right back! To improve initial learning of reset value
            early_start_policy = ResetPolicy(
                self.env, Policy(lambda _: self.env.action_space.sample())
            )
            early_start_policy.enable_evaluation()
            self.collector.set_policy(early_start_policy)
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
        self.rmv.add_tensor_batch(batch.s_1)

    # ==============================
    # Key Algorithm: Teleportation
    # ==============================
    def _reset(self):
        # Collector actually already resets the policy, so don't need to repeat here
        self.logger.log(
            {
                "reset_step": self.teleport_memory.episode_step,
                "reset_obs": self.collector.obs,
            }
        )
        if self.teleport_cfg.enable:
            # s_t = self.teleport_memory.targets()
            # s_r = self.reset_memory.targets()
            # s = torch.cat((s_t, s_r))
            # tid = self.teleport_strategy.select(s)
            # if tid >= len(s_t):
            #     self.teleport_strategy.select(s)
            # else:
            #     self.reset_memory.select(s)

            s_t = self.teleport_memory.targets()
            tid = self.teleport_strategy.select(s_t)
            self.teleport_memory.select(tid, self.collector)

            self.logger.log(
                {
                    "teleport_targets_index": tid,
                    "teleport_targets_observations": self.collector.obs,
                }
            )
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
            s_0_c, a_c, r_c, s_1_c, d_c, t_c = self.collector.collect(n=1)[-1]
            self._update_memory(s_0_c, a_c, r_c, s_1_c, d_c, t_c)
            self.rmv.add_tensor_batch(
                torch.tensor(s_1_c, device=self.device).unsqueeze(0)
            )

            # Teleport
            self.teleport_memory.update(env=self.collector.env, obs=s_0_c)

            if d_c or t_c:
                self._reset()

            # Updates
            batch, aux = self.memory.sample(self.cfg.train.minibatch_size)

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
            # Question: Should policy learn the value of resetting?
            if self.death_is_not_the_end or self.reset_as_an_action.enable:
                reset_sample = self.reset_memory.targets()
                c_1 = self.algorithm.critics.sample_network()
                c_2 = self.algorithm.critics.sample_network()
                a_1 = self.algorithm.policy_fn(reset_sample, critic=c_1.target)
                a_2 = self.algorithm.policy_fn(reset_sample, critic=c_2.target)
                target_max_1 = c_1.target.q(reset_sample, a_1).squeeze()
                target_max_2 = c_2.target.q(reset_sample, a_2).squeeze()
                reset_value = torch.minimum(target_max_1, target_max_2).mean().item()
                aux.v_1 = aux.v_1 * reset_value
                if self.reset_as_an_action.enable:
                    select = torch.logical_or(batch.t, batch.d)
                elif self.death_is_not_the_end:
                    select = batch.d
                aux.reset_value_mixin_select = select
                aux.reset_value_mixin_enable = True
                # Since we don't consider extrinsic rewards (for now)
                # Manually add the penalty to intrinsic rewards
                if self.reset_as_an_action:
                    r_i = r_i - batch.t * self.reset_as_an_action.penalty

            if self.death_is_not_the_end:
                batch.d = torch.zeros_like(batch.d, device=self.device).bool()

            batch.r = r_i
            self.algorithm.update(batch, aux, step=step)

            # Evaluation Epoch
            if step % self.cfg.log.frames_per_epoch == 0:
                self.logger.epoch()
                if self.death_is_not_the_end or self.reset_as_an_action:
                    self.logger.log({"reset_value": reset_value})

        # Store output
        self.logger.close()
