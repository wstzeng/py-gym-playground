import torch.nn as nn
from torch.distributions import Categorical
from .base_policy import BasePolicy

class ValuePolicy(BasePolicy):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.net = network

    def forward(self, x):
        return self.net(x)
