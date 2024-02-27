import pytest

import torch
import torch.nn as nn

from kitten.intrinsic.icm import IntrinsicCuriosityModule
from kitten.experience import Transition, AuxiliaryMemoryData

DEVICE = 'cpu'
torch.manual_seed(0)

class TestIntrinsicCuriosityModule:
    def build_networks(self, in_features=5, encoding_size=3, action_size=2):
        feature_net = nn.Sequential(
            nn.Linear(in_features=in_features,
                      out_features=encoding_size,
                      device=DEVICE,
                      dtype=torch.float32
            ),
            nn.LeakyReLU()
        )
        forward_head = nn.Linear(
            in_features=encoding_size+action_size,
            out_features=encoding_size,
            device=DEVICE,
            dtype=torch.float32
        )
        inverse_head = nn.Linear(in_features=encoding_size * 2, out_features=action_size)
        return feature_net, forward_head, inverse_head
    
    @pytest.mark.parametrize("batch_size", [(),(2,),(3,5)])
    @pytest.mark.parametrize("obs_size", [1,2,3])
    @pytest.mark.parametrize("encoding_size", [1,2,32])
    @pytest.mark.parametrize("action_size", [1,2])
    def test_icm(self,
                 batch_size: torch.Size,
                 obs_size: int,
                 encoding_size: int,
                 action_size: int):
        # Construction of ICM 
        feature_net, forward_head, inverse_head = self.build_networks(
            in_features=obs_size,
            encoding_size=encoding_size,
            action_size=action_size
        )

        icm = IntrinsicCuriosityModule(
            feature_net=feature_net,
            forward_head=forward_head,
            inverse_head=inverse_head
        )
        # Create fake data
        s_0 = torch.rand((*batch_size, obs_size))
        s_1 = torch.rand((*batch_size, obs_size))
        a = torch.rand((*batch_size, action_size))

        # Evaluate ICM
        phi_0, phi_1 = icm.feature_net(s_0), icm.feature_net(s_1)
        pred_phi_1 = icm.forward_model(s_0, a)
        pred_a = icm.inverse_model(s_0, s_1)

        assert phi_0.shape == ((*batch_size, encoding_size)), f"Expected encoding shape {(*batch_size, encoding_size)} but got {phi_0.shape}"
        assert pred_phi_1.shape == phi_1.shape, f"Shape of predicted encoding {pred_phi_1.shape} does not match true encoding {phi_1.shape}"
        assert pred_a.shape == a.shape, f"Shape of predicted action {pred_a.shape} does not match true action {a.shape}"
        
        # Reward
        r_i = icm(s_0, s_1, a)
        assert r_i.shape == batch_size, f"Reward shape {r_i.shape} should match batch size {batch_size}"

        # Loss
        aux = AuxiliaryMemoryData(torch.ones(batch_size), None, None)
        icm.update(Transition(s_0, a, None, s_1, None), aux, step=0)