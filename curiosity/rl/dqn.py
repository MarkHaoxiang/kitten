from typing import Dict
import torch
import torch.nn as nn
from torch import Tensor

from curiosity.curiosity.experience import AuxiliaryMemoryData, Transition
from curiosity.rl.rl import Algorithm

from curiosity.nn import AddTargetNetwork

# Note: This algorithm has been recently transferred from deprecated script form and is not yet functional

class DQN(Algorithm):
    """ Implements DQN

    V Mnih, et al. Playing Atari with Deep Reinforcement Learning. 2013.
    """
    def __init__(self,
                 critic: nn.Module,
                 gamma: float = 0.99,
                 lr: float = 1e-3,
                 update_frequency = 1,
                 device: str = "cpu"):
        self.critic = AddTargetNetwork(critic, device=device)

        self._gamma = gamma
        self._optim = torch.optim.Adam(params=self.critic.parameters(), lr=lr)
        self._update_frequency = update_frequency

        self.loss_critic_value = 0
    
    def td_error(self, s_0: Tensor, a: Tensor, r: Tensor, s_1: Tensor, d: Tensor):
        """ Returns TD difference for a transition
        """
        x = self.critic(s_0)[torch.arange(len(s_0), a)]
        with torch.no_grad():
            target_max = (~d) * torch.max(self.critic.target(s_1), dim=1).values
            y = r + target_max * self._gamma

        return y-x

    def update(self, batch: Transition, aux: AuxiliaryMemoryData, step: int):
        if step % self._update_frequency == 0:
            self._optim.zero_grad()
            loss = torch.mean((self.td_error(*batch) * aux.weights)**2)
            self.loss_critic_value = loss.item()
            loss.backward()
            self._optim.step()

        return self.loss_critic_value

    def get_log(self) -> Dict:
        return {
            "critic_loss": self.loss_critic_value
        }
