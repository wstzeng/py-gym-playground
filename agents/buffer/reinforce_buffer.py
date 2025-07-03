from .base_buffer import BaseBuffer

class ReinforceBuffer(BaseBuffer):
    def __init__(self):
        self.episodes = []

    def start_episode(self):
        self._log_probs = []
        self._rewards = []

    def record(self, log_prob, reward):
        self._log_probs.append(log_prob)
        self._rewards.append(reward)

    def end_episode(self):
        self.episodes.append((self._log_probs, self._rewards))

    def get_all(self):
        return self.episodes

    def clear(self):
        self.episodes.clear()
