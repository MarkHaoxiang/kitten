from typing import Callable, Optional, Union, List, Tuple
import random

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import gymnasium as gym

from kitten.experience import Transition
from kitten.rl import Algorithm
from kitten.rl.qt_opt import cross_entropy_method
from kitten.experience import AuxiliaryMemoryData
from kitten.nn import Critic, AddTargetNetwork

class QTOptCats(Algorithm):
    """ QtOpt with Ensembles
    """
    def __init__(self,
                 build_critic: Callable[[], Critic],
                 obs_space: gym.spaces.Box,
                 action_space: gym.spaces.Box,
                 ensemble_number: int = 5,
                 gamma: float = 0.99,
                 lr: float = 1e-3,
                 tau: float = 0.005,
                 cem_n: int = 64,
                 cem_m: int = 6,
                 cem_n_iterations: int = 2,
                 clip_grad_norm: Optional[float] = 1,
                 update_frequency: int = 1,
                 device: str = "cpu",
                 **kwargs) -> None:
        super().__init__()

        self.critics = [AddTargetNetwork(build_critic(), device=device) for _ in range(ensemble_number)]

        self.device=device

        self._ensemble_number = ensemble_number
        self._gamma = gamma
        self._tau = tau
        self._env_action_scale = torch.tensor(action_space.high-action_space.low, device=device) / 2.0
        self._env_action_min = torch.tensor(action_space.low, dtype=torch.float32, device=device)
        self._env_action_max = torch.tensor(action_space.low, dtype=torch.float32, device=device)
        self._obs_space = obs_space
        self._action_space = action_space
        self._clip_grad_norm = clip_grad_norm
        self._update_frequency = update_frequency
        self._n = cem_n
        self._m = cem_m
        self._n_iterations = cem_n_iterations
        self._chosen_critic = 0

        self._optim_critic = torch.optim.Adam(
            params=torch.nn.ModuleList(self.critics).parameters(),
            lr=lr
        )

        self.loss_critic_value = 0
    
    @property
    def critic(self):
        return self.critics[self._chosen_critic]
    
    def mu_var(self, s: Tensor, a: Tensor):
        """ Calculates expectation and epistemic uncertainty of a state-action pair
        """
        q = np.array([critic.q(s, a) for critic in self.critics])
        mu, var = q.mean(), q.var()
        return mu, var


    def policy_fn(self, s: Union[Tensor, np.ndarray], critic: Optional[Critic] = None) -> Tensor:
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, device=self.device)
        if critic is None:
            critic = self.critic
        squeeze = False
        if self._obs_space.shape == s.shape:
            squeeze = True
            s = s.unsqueeze(0)
        result = cross_entropy_method(
            s_0=s,
            critic_network=critic,
            action_space=self._action_space,
            n=self._n,
            m=self._m,
            n_iterations=self._n_iterations,
            device=self.device
        )
        if squeeze:
            result = result.squeeze()
        return result

    def reset_critic(self):
        self._chosen_critic = random.randint(0,self._ensemble_number-1)

    def _critic_update(self, batch: Transition, aux: AuxiliaryMemoryData):
        x = [critic.q(batch.s_0, batch.a).squeeze() for critic in self.critics]        
        with torch.no_grad():
            # Implement a variant of Clipped Double-Q Learning
            # Randomly sample two networks
            sampled_critics = random.sample(self.critics, 2)
            a_1 = self.policy_fn(batch.s_1, critic=sampled_critics[0].target)
            a_2 = self.policy_fn(batch.s_1, critic=sampled_critics[1].target)
            target_max_1 = sampled_critics[0].target.q(batch.s_1, a_1).squeeze()
            target_max_2 = sampled_critics[1].target.q(batch.s_1, a_2).squeeze()
            y = batch.r + (~batch.d) * torch.minimum(target_max_1, target_max_2) * self._gamma

        losses = []
        for x_i in x:
            losses.append(torch.mean((aux.weights * (y-x_i)) ** 2))
        loss_critic = sum(losses)

        loss_value = loss_critic.item()
        self._optim_critic.zero_grad()
        loss_critic.backward()
        if not self._clip_grad_norm is None:
            for critic in self.critics:
                nn.utils.clip_grad_norm_(critic.net.parameters(), self._clip_grad_norm)
        self._optim_critic.step()

        return loss_value

    def update(self, batch: Transition, aux: AuxiliaryMemoryData, step: int):
        if step % self._update_frequency == 0:
            self.loss_critic_value =  self._critic_update(batch, aux)
            for critic in self.critics:
                critic.update_target_network(tau = self._tau)
        return self.loss_critic_value

    def get_log(self):
        return {
            "critic_loss": self.loss_critic_value,
        }

    def get_models(self) -> List[Tuple[nn.Module, str]]:
        return list(zip(self.critics, [f"critic_{i}" for i in range(len(self.critics))]))
 