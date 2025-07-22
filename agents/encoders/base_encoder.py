from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseEncoder(ABC):
    """
    Abstract base class for all state encoders.
    Defines the common interface for encoding environment observations.
    """
    def __init__(self, state_dim):
        self.state_dim = state_dim

    @abstractmethod
    def encode(self, observation) -> torch.Tensor:
        """
        Encodes a raw observation from the environment into a tensor
        suitable for neural network processing.

        Args:
            observation: The raw observation from the gymnasium environment.

        Returns:
            torch.Tensor: The encoded state as a PyTorch tensor.
        """
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """
        Returns the dimension of the encoded output.
        """
        pass
