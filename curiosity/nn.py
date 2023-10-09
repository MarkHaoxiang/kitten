import copy

from gymnasium import Env
from gymnasium.spaces import Box, Discrete
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
class GaussianActor(nn.Module):
    """ Implements a policy module with normal noise applied, for continuous environments.
    """
    def __init__(self, mean_network: nn.Module, std_network: nn.Module):
        """ Initialises the gaussian actor

        Args:
            mean_network (nn.Module): NN from observations to the mean.
            std_network (nn.Module): NN from observations to std.
        """
        super().__init__()
        self.mean = mean_network
        self.std = std_network

    def forward(self, x: torch.Tensor, noise: bool = True) -> torch.Tensor:
        """ Calculates the action to take

        Args:
            x (torch.Tensor): obs
            noise (bool, optional): Toggle to add gaussian noise.
                For example, set to false when evaluating.
                Defaults to True.

        Returns:
            torch.Tensor: action
        """
        action = self.mean(x)
        if noise:
            std = self.std(x)
            action = torch.normal(action, std)
        return action
    
class ClassicalGaussianActor(GaussianActor):
    """ An actor for continuous low-dimensionality gym environments
    """
    def __init__(self, features: int, n_actions: int):
        shared = nn.Sequential(
            nn.LazyLinear(out_features=features),
            nn.Tanh(),
            nn.Linear(in_features=features, out_features=features),
            nn.Tanh(),
        )
        mean = nn.Sequential(
            shared,
            nn.Linear(in_features=features, out_features=n_actions)
        )
        std = nn.Sequential(
            shared,
            nn.Linear(in_features=features, out_features=n_actions),
            nn.Sigmoid()
        )
        super(mean, std)

# TODO(mark)
# - AtariGaussianActor

def build_critic(env: Env, features: int):
    """ Build a default critic network for env on a single-agent environment
    """
    # Observation space
    # Action space
    discrete = isinstance(env.action_space, Discrete)
    if not discrete and len(env.action_space.shape) != 1:
        raise ValueError(f"Action space {env.action_space} is not supported")

    # Observation space
    if not isinstance(env.observation_space, Box):
        raise ValueError(f"Observation space {env.observation_space} is not supported")
    pixels = len(env.observation_space.shape) != 1

    if not pixels:
        if discrete:
            return nn.Sequential(
                nn.Linear(in_features=env.observation_space.shape[-1] + (0 if discrete else len(env.action_space.shape)), out_features=features),
                nn.Tanh(),
                nn.Linear(in_features=features, out_features=features),
                nn.Tanh(),
                nn.Linear(in_features=features, out_features=env.action_space.n if discrete else 1)
            )
    else:
        raise NotImplementedError("Pixel space is a work in progress")

# TODO(mark)
# - Whats a good method for sharing layers between actor and critic. Is it even neccessary?
