import copy

import torch.nn as nn

class AddTargetNetwork(nn.Module):
    """ Wraps an extra target network for delayed updates

    Args:
        nn (_type_): Original network to duplicate
    """
    def __init__(self, net: nn.Module, device = "cpu"):
        super().__init__()
        self.net = net
        self.target = copy.deepcopy(net)
        self.net = self.net.to(device)
        self.target = self.target.to(device)
    
    def forward(self, x):
        return self.net(x)

    def update_target_network(self, tau: float) -> None:
        assert (0 <= tau <= 1), f"Weighting {tau} is out of range"
        for target_param, net_param in zip(self.target.parameters(), self.net.parameters()):
            target_param.data.copy_(net_param.data * tau + target_param.data*(1-tau))
