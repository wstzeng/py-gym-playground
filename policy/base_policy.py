import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BasePolicyNetwork(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, state):
        ...

    @abstractmethod
    def distribution(self, state):
        """Return a torch.distributions.Distribution object"""
        ...

    def select_action(self, state, action=None):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        else:
            state = state.float()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(next(self.parameters()).device)
        dist = self.distribution(state)

        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)

        if isinstance(action, torch.Tensor):
            action = action.squeeze(0)

        if isinstance(log_prob, torch.Tensor) and log_prob.ndim > 0:
            log_prob = log_prob.sum(axis=-1)  # for multivariate Normal, etc.

        return action.detach().cpu().numpy() if not isinstance(action, (int, float)) else action.item(), log_prob
