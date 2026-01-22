# agents/base_agent.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseAgent(ABC):
    def __init__(self, encoder: nn.Module, device: str = "auto"):
        self.encoder = encoder
        # Device management
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.encoder.to(self.device)

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def record(self, info: dict, reward: float, done: bool, **kwargs):
        pass

    @abstractmethod
    def update(self):
        pass
    
    def save_checkpoints(self, path: str):
        """
        Generic save method that captures the state of all core components.
        """
        import os
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        checkpoint = {
            'encoder': self.encoder.state_dict(),
        }
        
        if hasattr(self, 'policy'):
            checkpoint['policy'] = self.policy.state_dict()
        if hasattr(self, 'critic'):
            checkpoint['critic'] = self.critic.state_dict()
        if hasattr(self, 'optimizer'):
            checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)
        print(f"[*] Checkpoint saved to {path}")

    def load_checkpoints(self, path: str):
        """Generic load method."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        
        if hasattr(self, 'policy') and 'policy' in checkpoint:
            self.policy.load_state_dict(checkpoint['policy'])
        print(f"[*] Checkpoint loaded from {path}")
