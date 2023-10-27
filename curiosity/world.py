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
                 eta: float = 0.01,
                 beta: float = 0.2,
                 discrete_action_space: bool = False):
        """ Creates the intrinsic curiosity module

        Args:
            feature_net (nn.Module): From observations to encodings
            forward_head (nn.Module): From encoding and action to encoding
            inverse_head (nn.Module): From two encodings to action
            eta (float, optional): Intrinsic reward scaling factor. Default 0.01.
            beta (float, optional): Forward/Inverse loss scaling factor. Default 0.2.
        """
        super().__init__()
        self.feature_net = feature_net
        self.forward_head = forward_head
        self.inverse_head = inverse_head
        self.eta = eta
        self.beta = beta
        self.discrete = discrete_action_space
        if self.discrete:
            raise NotImplementedError("Not yet implemented for discrete action space")

        self._mse_loss = nn.MSELoss()

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
    
    def calc_loss(self, s_0: torch.Tensor, s_1: torch.Tensor, a: torch.Tensor):
        pred_phi_1 = self.forward_model(s_0, a)
        true_phi_1 = self.feature_net(s_1)
        forward_loss = 0.5 * self._mse_loss(true_phi_1, pred_phi_1)
        pred_a = self.inverse_model(s_0, s_1)
        if self.discrete:
            raise NotImplementedError("Not yet implemented for discrete action space")
        else:
            inverse_loss = self._mse_loss(a, pred_a)
        return forward_loss * self.beta + inverse_loss * (1-self.beta)

    # TODO(mark)
    # Expand to support general gym environment
    @staticmethod
    def build(obs_size: int,
              encoding_size: int,
              action_size: int,
              device: str ="cpu",
              **kwargs):
        """ Builds a default intrinsic curiosity module

        Args:
            obs_size (int): Observation shape
            encoding_size (int): Number of features for the encoding
            action_size (int): Action shape
            device (str, optional): Tensor device. Defaults to "cpu".

        Returns:
            _type_: ICM
        """
        feature_net = nn.Sequential(
            nn.Linear(in_features=obs_size, out_features=encoding_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=encoding_size, out_features=encoding_size),
            nn.LeakyReLU()
        ).to(device=device)

        forward_head = nn.Sequential(
            nn.Linear(in_features=encoding_size+action_size, out_features=encoding_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=encoding_size, out_features=encoding_size)
        ).to(device=device)

        inverse_head = nn.Sequential(
            nn.Linear(in_features=encoding_size * 2, out_features=encoding_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=encoding_size, out_features=action_size)
        ).to(device=device)
 
        return IntrinsicCuriosityModule(feature_net, forward_head, inverse_head, discrete_action_space=False, **kwargs)
