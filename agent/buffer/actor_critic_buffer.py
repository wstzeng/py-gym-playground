from .base_buffer import BaseBuffer

class ActorCriticBuffer(BaseBuffer):
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def store(self, log_prob, value, reward, done):
        """
        Stores step data. 
        'value' is the state value estimate from the critic head.
        """
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def get_data(self):
        """
        Returns all collected data for update logic.
        """
        return self.log_probs, self.values, self.rewards, self.dones

    def clear(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
