import copy
from typing import TypeVar, Generic

from torch import nn


T = TypeVar("T", bound=nn.Module)


class AddTargetNetwork(nn.Module, Generic[T]):
    """Wraps an extra target network for delayed updates"""

    def __init__(self, net: T, device="cpu"):
        """Wraps an extra target network for delayed updates

        Args:
            net (T): Network.
            device (str, optional): Network device. Defaults to "cpu".
        """
        super().__init__()
        self._net = net
        self._target = copy.deepcopy(net)
        self._net = self.net.to(device)
        self._target = self.target.to(device)

    def forward(self, x, *args, **kwargs):
        return self.net(x, *args, **kwargs)

    def update_target_network(self, tau: float = 1) -> None:
        """Updates target network to follow latest parameters

        Args:
            tau (float): Weighting factor. Defaults to 1.
        """
        assert 0 <= tau <= 1, f"Weighting {tau} is out of range"
        for target_param, net_param in zip(
            self.target.parameters(), self.net.parameters()
        ):
            target_param.data.copy_(
                net_param.data * tau + target_param.data * (1 - tau)
            )

    @property
    def net(self) -> T:
        return self._net

    @property
    def target(self) -> T:
        return self._target

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return self.net.__getattribute__(name)
