import torch

class ReplayBuffer:
    """ Store and access traces of episodes
    """
    def __init__(self,
                 capacity: int,
                 shape: int,
                 device: str = "cpu"):
        self.capacity = capacity
        self.shape = shape
        self.device = device

        self.storage = torch.zeros(size = (self.capacity, self.shape), device=self.device)
