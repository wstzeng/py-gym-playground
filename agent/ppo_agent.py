import torch
import torch.nn.functional as F
import torch.optim as optim
from . import BaseAgent
from .buffer.ppo_buffer import PPOBuffer


class PPOAgent(BaseAgent):
    def __init__(self, policy, value_net, lr=3e-4, gamma=0.99, eps_clip=0.2, device='cpu'):
        super().__init__()
        self.policy = policy.to(device)
        self.value_net = value_net.to(device)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.device = device
        self.buffer = PPOBuffer()
