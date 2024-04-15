import torch

from kitten.experience import Transitions
from kitten.nn import Value

def monte_carlo_return(batch: Transitions,
                   gamma: float,
                   value: Value | None = None
):
    """ Returns the episode return for batch. If final

    Args:
        batch (Transitions): Collected episode.
        value (Value | None): Used as target if final value is truncated. Value bootstrap.
    """
    # TODO: Deal with different batch shapes
    value_targets = [0 for _ in range(len(batch))]
    value_targets[-1] = batch.r[-1]
    if not batch.d[-1]:
        assert value is not None, "Episode not done and no Value provided"
        value_targets[-1] += gamma * value.v(batch.s_1[-1])
    for i in range(1, len(batch)):
        value_targets[-i-1] = batch.r[-i-1] + gamma * value_targets[-i]
    return torch.tensor(value_targets, device=batch.s_0.get_device())

def generalised_advantage_estimation(batch: Transitions,
                                     lmbda: float,
                                     gamma: float,
                                     value: Value
):
    with torch.no_grad():
        n = len(batch.r)
        delta = (
            batch.r
            - value.v(batch.s_0)
            + gamma * value.v(batch.s_1) * (~batch.d)
        )
        f = gamma * lmbda
        advantages = torch.zeros_like(batch.r, device=batch.r.get_device())
        advantages[-1] = delta[-1]
        for i in range(1, n):
            advantages[-(i + 1)] = f * advantages[-i] + delta[-(i + 1)]
        return advantages

def td_lambda(batch: Transitions,
              lmbda: float,
              gamma: float,
              value: Value):
    return monte_carlo_return(batch, gamma, value) + generalised_advantage_estimation(batch, lmbda, gamma, value)

