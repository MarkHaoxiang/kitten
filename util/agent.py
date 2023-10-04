from torch import nn
from torch import Tensor

class ActorCritic(nn.Module):
    """ Defines an actor critic agent
    """
    def __init__(self, actor: nn.Module, critic: nn.Module):
        super().__init__(self)
        self.actor_net = actor
        self.critic_net = critic

    def forward(self, x: Tensor, option="actor") -> Tensor:
        match option:
            case "actor":
                return self.actor(x)
            case "critic":
                return self.critic(x)
            case _:
                f"Unsupported option {option}"

    def actor(self, x: Tensor) -> Tensor:
        return self.actor_net(x)
 
    def critic(self, x: Tensor) -> Tensor:
        return self.critic_net(x)

class RandomActor(ActorCritic):
    """Returns random values to be used in injection""" 
    # TODO(mark)
    pass