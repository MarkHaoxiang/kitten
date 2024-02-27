import torch
import torch.nn as nn
from torch.optim import SGD

from kitten.nn import AddTargetNetwork

DEVICE = 'cpu'
torch.manual_seed(0)

class TestTargetNetwork:
    def build_test_network(self):
        return nn.Linear(in_features=3, out_features=1)

    def test_target_network(self):
        # Test creation
        net = AddTargetNetwork(self.build_test_network())
        test_data = torch.rand((3,))
        assert net(test_data).shape == net.target(test_data).shape, "Shape of target network and network are not equal"
        # Test Optimise
        original_net = net(test_data)
        original_target = net.target(test_data)
        optim = SGD(net.net.parameters(), lr=0.01)
        optim.zero_grad()
        original_net.mean().backward()
        optim.step()
        new_net = net(test_data)
        new_target = net.target(test_data)
        assert new_net != original_net
        assert new_target == original_target
        # Test update
        net.update_target_network(1.0)
        assert net(test_data) == net.target(test_data)
