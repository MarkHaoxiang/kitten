import math
from typing import Any

import numpy as np
import torch

from kitten.nn import Actor, StochasticActor, HasActor
from kitten.experience import Transitions
from kitten.common.rng import Generator
from kitten.rl.util import generate_minibatches

from .interface import Algorithm, AuxiliaryData
from .advantage import AdvantageEstimator

class ProximalPolicyOptimisation(Algorithm[AuxiliaryData], HasActor):
    def __init__(
        self,
        actor: StochasticActor,
        advantage_estimation: AdvantageEstimator,
        rng: Generator,
        update_epochs: int = 4,
        minibatch_size: int = 32,
        clip_ratio: float = 0.1,
        lr: float = 1e-3,
        **kwargs,
    ) -> None:
        self._actor = actor
        self._advantage_estimation = advantage_estimation
        self._update_epochs = update_epochs
        self._minibatch_size = minibatch_size
        self._clip_ratio = clip_ratio
        self._optim = torch.optim.Adam(params=self._actor.parameters(), lr=lr)
        self._rng = rng.numpy
        self._loss = torch.nn.MSELoss()

    def update(self, batch: Transitions, aux: AuxiliaryData, step: int) -> Any:
        # Calculate Advantage
        a_hat = self._advantage_estimation.A(batch)
        # Update Policy
        with torch.no_grad():
            log_prob_orig = self._actor.log_prob(batch.s_0, batch.a)
        total_loss = 0
        for _ in range(self._update_epochs):
            for i, mb in generate_minibatches(batch, mb_size=self._minibatch_size, rng=self._rng):
                self._optim.zero_grad()
                a_hat_mb = a_hat[i]
                ratio = (self._actor.log_prob(mb.s_0, mb.a)-log_prob_orig[i]).exp().squeeze()
                actor_loss = -torch.minimum(
                    ratio * a_hat_mb,
                    torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
                    * a_hat_mb,
                ).mean()
                total_loss += actor_loss.item()
                actor_loss.backward()
                self._optim.step()

        return total_loss

    @property
    def actor(self) -> Actor:
        return self._actor

    def policy_fn(self, s: torch.Tensor) -> torch.Tensor:
        return self.actor.a(s)
