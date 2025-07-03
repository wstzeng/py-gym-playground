import torch
from torch import (nn as nn, optim as optim)
from . import BaseAgent
from .buffer.reinforce_buffer import ReinforceBuffer

class ReinforceAgent(BaseAgent):
    def __init__(self, policy, lr=1e-3, gamma=0.99, device='cpu'):
        super().__init__()
        self.policy = policy.to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device
        self.buffer = ReinforceBuffer()

    def start_episode(self):
        self.buffer.start_episode()

    def record_reward(self, reward):
        self.buffer.record(self._latest_log_prob, reward)

    def end_episode(self):
        self.buffer.end_episode()

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action, log_prob = self.policy.select_action(state)
        self._latest_log_prob = log_prob
        return action

    def update_policy(self):
        all_returns = []
        all_log_probs = []

        for log_probs, rewards in self.buffer.get_all():
            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            all_returns.extend(returns)
            all_log_probs.extend(log_probs)

        returns = torch.tensor(all_returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -torch.sum(torch.stack([
            log_prob * ret for log_prob, ret in zip(all_log_probs, returns)
        ]))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.buffer.clear()
        return loss.item()
