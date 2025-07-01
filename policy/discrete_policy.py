from torch.distributions import Categorical
from .base_policy import BasePolicyNetwork
import torch.nn as nn
import torch.nn.functional as F

class DiscretePolicyNetwork(BasePolicyNetwork):
    def __init__(self, state_dim, action_dim, hidden_sizes=[128, 128]):
        super().__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.LeakyReLU())
            input_dim = h
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, action_dim)

    def forward(self, state):
        x = self.feature_extractor(state)
        logits = self.output_layer(x)
        return F.softmax(logits, dim=-1)

    def distribution(self, state):
        probs = self.forward(state)
        return Categorical(probs)
