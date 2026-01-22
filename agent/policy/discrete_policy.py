import torch.nn as nn
from torch.distributions import Categorical
from .base_policy import BasePolicy

class DiscretePolicy(BasePolicy):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.net = network

    def forward(self, x):
        return self.net(x)

    def get_distribution(self, x):
        logits = self.forward(x)
        return Categorical(logits=logits)
 