import copy
from typing import Union

from gymnasium import Env
import torch
import torch.nn as nn

class AddTargetNetwork(nn.Module):
    """ Wraps an extra target network for delayed updates
    """
    def __init__(self, net: nn.Module, device = "cpu"):
        super().__init__()
        self.net = net
        self.target = copy.deepcopy(net)
        self.net = self.net.to(device)
        self.target = self.target.to(device)
    
    def forward(self, x, *args, **kwargs):
        return self.net(x, *args, **kwargs)

    def update_target_network(self, tau: float = 1) -> None:
        """ Updates target network to follow latest parameters

        Args:
            tau (float): Weighting factor. Defaults to 1.
        """
        assert (0 <= tau <= 1), f"Weighting {tau} is out of range"
        for target_param, net_param in zip(self.target.parameters(), self.net.parameters()):
            target_param.data.copy_(net_param.data * tau + target_param.data*(1-tau))

class ClassicalBoxActor(nn.Module):
    """ Actor for continuous low-dimensionality gym environments
    """
    def __init__(self, env: Env, features=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=env.observation_space.shape[0], out_features=features),
            nn.LeakyReLU(),
            nn.Linear(in_features=features, out_features=features),
            nn.LeakyReLU(),
            nn.Linear(in_features=features, out_features=env.action_space.shape[-1]),
            nn.Tanh()
        )
        self.register_buffer("scale", torch.tensor(env.action_space.high-env.action_space.low) / 2.0)
        self.register_buffer("bias", torch.tensor(env.action_space.high+env.action_space.low) / 2.0)

    def forward(self, x: torch.Tensor):
        return self.net(x) * self.scale + self.bias
