from abc import ABC, abstractmethod
import torch

class BaseAgent(ABC):
    def __init__(self):
        return

    def start_episode(self):
        self.buffer.start_episode()

    def record_log_prob(self, log_prob):
        self.buffer.record(log_prob=log_prob)

    def record_reward(self, reward):
        self.buffer.record(reward=reward)

    def end_episode(self):
        self.buffer.end_episode()

    def clear_buffer(self):
        self.buffer.clear()

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def update_policy(self):
        pass

    def save_policy(self, fname: str):
        if not hasattr(self, 'policy'):
            raise AttributeError('Agent has no policy attribute to save.')
        torch.save(self.policy.state_dict(), fname)

    def load_policy(self, fname: str):
        if not hasattr(self, 'policy'):
            raise AttributeError('Agent has no policy attribute to load.')
        self.policy.load_state_dict(torch.load(fname))
