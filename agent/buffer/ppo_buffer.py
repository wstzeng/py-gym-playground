import torch
from .base_buffer import BaseBuffer

class PPOBuffer(BaseBuffer):
    def __init__(self):
        # Buffer initialization for PPO trajectory data
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def store(self, info, reward, done):
        # Stores step data including state and action from info dict
        self.states.append(info['state'])
        self.actions.append(info['action'])
        self.log_probs.append(info['log_prob'])
        self.values.append(info['value'])
        self.rewards.append(reward)
        self.dones.append(done)

    def get_data(self):
        # Returns all trajectory data for PPO update logic
        return (
            self.states, 
            self.actions, 
            self.log_probs, 
            self.values, 
            self.rewards, 
            self.dones
        )

    def clear(self):
        # Clears all lists for the next rollout session
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
