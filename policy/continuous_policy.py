from torch.distributions import Normal
from .base_policy import BasePolicyNetwork
import torch.nn as nn
import torch
import torch.nn.functional as F

class ContinuousPolicyNetwork(BasePolicyNetwork):
    def __init__(self, state_dim, action_dim, hidden_sizes=[128, 128]):
        super().__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.LeakyReLU())
            input_dim = h
        self.feature_extractor = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.feature_extractor(state)
        mean = self.mean_layer(x)
        return mean

    def distribution(self, state):
        mean = self.forward(state)
        std = torch.exp(self.log_std)
        return Normal(mean, std)
