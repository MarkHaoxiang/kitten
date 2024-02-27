from typing import Dict, List, Tuple, Union
from numpy import ndarray
import torch
import torch.nn as nn
from torch import Tensor

from kitten.experience import AuxiliaryMemoryData, Transition
from kitten.rl import Algorithm, HasCritic

from kitten.nn import AddTargetNetwork, Critic


class DQN(Algorithm, HasCritic):
    """Implements DQN

    V Mnih, et al. Playing Atari with Deep Reinforcement Learning. 2013.
    """

    def __init__(
        self,
        critic: Critic,
        gamma: float = 0.99,
        lr: float = 1e-3,
        update_frequency=1,
        target_update_frequency: int = 100,
        device: str = "cpu",
        **kwargs,
    ):
        """Deep Q-Networks

        Args:
            critic (Critic): Discrete critic.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            lr (float, optional): Critic learning rate. Defaults to 1e-3.
            update_frequency (int, optional): _description_. Defaults to 1.
            target_update_frequency (int, optional): _description_. Defaults to 100.
            device (str, optional): _description_. Defaults to "cpu".
        """
        self.device = device

        self._critic = AddTargetNetwork(critic, device=device)
        self._gamma = gamma
        self._optim = torch.optim.Adam(params=self.critic.parameters(), lr=lr)
        self._update_frequency = update_frequency
        self._target_update_frequency = target_update_frequency
        self.loss_critic_value = 0

    @property
    def critic(self) -> Critic:
        return self._critic

    def td_error(self, s_0: Tensor, a: Tensor, r: Tensor, s_1: Tensor, d: Tensor):
        """Returns TD difference for a transition"""
        # TODO: This doesn't work well with multiple batch sizes
        x = self._critic.q(s_0)[torch.arange(len(s_0)), a]
        with torch.no_grad():
            target_max = (~d) * torch.max(self.critic.target.q(s_1), dim=1).values
            y = r + target_max * self._gamma
        return y - x

    def update(self, batch: Transition, aux: AuxiliaryMemoryData, step: int):
        if step % self._update_frequency == 0:
            self._optim.zero_grad()
            loss = torch.mean((self.td_error(*batch) * aux.weights) ** 2)
            self.loss_critic_value = loss.item()
            loss.backward()
            self._optim.step()
        if step % self._target_update_frequency == 0:
            self._critic.update_target_network()

        return self.loss_critic_value

    def policy_fn(self, s: Union[Tensor, ndarray]) -> Tensor:
        if isinstance(s, ndarray):
            s = torch.tensor(s, device=self.device, dtype=torch.float32)
        return torch.argmax(self.critic.q(s=s, a=None), dim=-1)

    def get_log(self) -> Dict:
        return {"critic_loss": self.loss_critic_value}

    def get_models(self) -> List[Tuple[nn.Module, str]]:
        return [(self.critic.net, "critic")]
