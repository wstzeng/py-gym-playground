# agents/buffer/reinforce_buffer.py
from .base_buffer import BaseBuffer

class ReinforceBuffer(BaseBuffer):
    def __init__(self):
        self.log_probs = []
        self.rewards = []

    def store(self, log_prob, reward):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def get_data(self):
        return self.log_probs, self.rewards

    def clear(self):
        self.log_probs = []
        self.rewards = []
