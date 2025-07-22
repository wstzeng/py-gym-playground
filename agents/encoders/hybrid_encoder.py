# agents/state_encoders/hybrid_encoder.py
import torch
from gymnasium.spaces import Space, Dict, Tuple
from typing import List, Union, Dict as DictType, Any
from .base_encoder import BaseEncoder

class HybridEncoder(BaseEncoder):
    """
    An encoder that handles multi-modal observations (e.g., gymnasium.spaces.Dict or Tuple).
    It accepts a list of BaseEncoder instances, each responsible for a specific part
    of the multi-modal observation, and concatenates their outputs.
    """
    def __init__(self, observation_space: Union[Dict, Tuple], encoders: List[BaseEncoder]):
        super().__init__(observation_space)

        if not isinstance(observation_space, (Dict, Tuple)):
            raise ValueError("HybridEncoder expects a gymnasium.spaces.Dict or Tuple observation_space.")

        if len(encoders) != len(observation_space.spaces):
            raise ValueError(
                f"Number of encoders ({len(encoders)}) must match the number of "
                f"sub-spaces in the observation_space ({len(observation_space.spaces)})."
            )

        self.encoders = torch.nn.ModuleList(encoders) # Use ModuleList if encoders have learnable params

        # Map encoders to sub-spaces (useful for Dict spaces)
        self.space_keys = None
        if isinstance(observation_space, Dict):
            self.space_keys = list(observation_space.spaces.keys())

        # Calculate total output dimension
        self._output_dim = sum(encoder.get_output_dim() for encoder in self.encoders)

    def encode(self, observation: Union[DictType[str, Any], Tuple[Any, ...]]) -> torch.Tensor:
        """
        Encodes a multi-modal observation by calling each sub-encoder and concatenating their outputs.

        Args:
            observation: The raw multi-modal observation from the gymnasium environment
                         (e.g., a dictionary for Dict space, or a tuple for Tuple space).

        Returns:
            torch.Tensor: The concatenated encoded state as a PyTorch tensor.
        """
        encoded_parts = []
        if isinstance(self.observation_space, Dict):
            # For Dict spaces, iterate through keys in order
            for i, key in enumerate(self.space_keys):
                sub_observation = observation[key]
                encoded_parts.append(self.encoders[i].encode(sub_observation))
        elif isinstance(self.observation_space, Tuple):
            # For Tuple spaces, iterate through elements
            for i, sub_observation in enumerate(observation):
                encoded_parts.append(self.encoders[i].encode(sub_observation))
        else:
            # Should not happen due to constructor check
            raise TypeError("Unsupported observation space type for HybridEncoder.")

        return torch.cat(encoded_parts, dim=-1) # Concatenate along the last dimension

    def get_output_dim(self) -> int:
        """
        Returns the total dimension of the concatenated encoded output.
        """
        return self._output_dim
