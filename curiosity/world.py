import torch
import torch.nn as nn

class IntrinsicCuriosityModule(nn.Module):
    """ Intrinsic curiosity, intrinsic reward

    https://arxiv.org/pdf/1705.05363.pdf
    """
    
    def __init__(self,
                 feature_net: nn.Module,
                 forward_head: nn.Module,
                 inverse_head: nn.Module,
                 eta = 0.01):
        """ Creates the intrinsic curiosity module

        Args:
            feature_net (nn.Module): From observations to encodings
            forward_head (nn.Module): From encoding and action to encoding
            inverse_head (nn.Module): From two encodings to action
            eta (float, optional): Intrinsic reward scaling factor. Default 0.01.
        """
        super().__init__()
        self.feature_net = feature_net
        self.forward_head = forward_head
        self.inverse_head = inverse_head
        self.eta = eta

    def forward_model(self, s_0: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """ Predicts next encoding

        Args:
            s_0 (torch.Tensor): Current obs
            a (_type_): Current action

        Returns:
            torch.Tensor: Next encoding
        """
        batch_size = s_0.shape[:-1]
        phi_0 = self.feature_net(s_0)
        pred_phi_1 = self.forward_head(torch.cat((phi_0, a), len(batch_size)))
        return pred_phi_1
    
    def inverse_model(self, s_0: torch.Tensor, s_1: torch.Tensor) -> torch.Tensor:
        """Predicts action

        Args:
            s_0 (torch.Tensor): Current obs
            s_1 (torch.Tensor): Next obs

        Returns:
            torch.Tensor: Action
        """
        batch_size = s_0.shape[:-1]
        phi_0 = self.feature_net(s_0)
        phi_1 = self.feature_net(s_1)
        a = self.inverse_head(torch.cat((phi_0, phi_1), len(batch_size)))
        return a

    def forward(self, s_0: torch.Tensor, s_1: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """ Calculates the intrinsic reward

        Args:
            s_0 (torch.Tensor): Current obs
            s_1 (torch.Tensor): Next obs
            a (torch.Tensor): Action

        Returns:
            torch.Tensor: Intrinsic reward
        """
        pred_phi_1 = self.forward_model(s_0, a)
        true_phi_1 = self.feature_net(s_1)
        r_i = torch.linalg.vector_norm(true_phi_1-pred_phi_1, dim=(-1)) * self.eta / 2
        return r_i
