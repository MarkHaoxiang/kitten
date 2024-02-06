import gymnasium as gym
import torch
import torch.nn as nn

from curiosity.rl import Algorithm

def cross_entropy_method(s_0: torch.Tensor, 
                         critic_network: nn.Module,
                         action_space: gym.spaces.Box,
                         n: int = 64,
                         m: int = 6,
                         n_iterations: int = 2,
                         device: str="cpu") -> torch.Tensor:
    """ Sampling method to find action for a Q function on continuous spaces

    Args:
        critic_network (torch.nn): Critic.
        action_space (gym.spaces.Box): Action space
        n (int, optional): Number of values each iteration.
        m (int, optional): Best values each iteration
        n_iterations (int, optional): Number of convergence iterations.
        device (str, optional): Device of critic and output.

    Returns:
        torch.Tensor: Action
    """
    if n_iterations <= 0:
        raise ValueError("Number of iterations must be at least 1, currently {n_iterations}")

    # For clipping
    env_action_min = torch.tensor(action_space.low, dtype=torch.float32, device=device)
    env_action_max = torch.tensor(action_space.high, dtype=torch.float32, device=device)

    # Reshape
    s_0 = s_0.unsqueeze(1).expand((s_0.shape[0],n,*s_0.shape[1:]))

    # Run cem
    sampled_actions = torch.rand((s_0.shape[0], n, *action_space.shape),device=device)
    sampled_actions = sampled_actions * (env_action_max-env_action_min)[None, :, None] \
        + env_action_min[None, :, None]
    batch_indices = torch.arange(sampled_actions.shape[0]).view(-1, 1).expand(-1, m)

    for _ in range(n_iterations):
        q = critic_network(torch.cat((s_0, sampled_actions), -1))
        indices = torch.topk(q, m, dim=1).indices.squeeze()
        best_actions = sampled_actions[batch_indices, indices, :]
        mu, std = best_actions.mean(dim=1), best_actions.std(dim=1)
        sampled_actions = torch.randn_like(sampled_actions) * std[:,None,:] + mu[:,None,:]
        sampled_actions = torch.clamp(
            sampled_actions,
            min=env_action_min,
            max=env_action_max
        )
    
    return mu

class QT_Opt(Algorithm):
    """ Q-learning for continuous actions with Cross-Entropy Maximisation

    A custom variant with TD3 critic improvements

    Kalashnikov et al. QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation
    """
    def __init__(self,
                 critic_network: nn.Module,
                 **kwargs) -> None:
        super().__init__()
