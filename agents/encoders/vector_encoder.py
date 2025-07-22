import torch
import torch.nn as nn
from .base_encoder import BaseEncoder

class VectorEncoder(BaseEncoder):
    def __init__(self, state_dim, output_dim=128, architecture=None):
        super().__init__(state_dim)

        if isinstance(architecture, nn.Module):
            self.architecture = architecture
        else:
            if output_dim is None:
                raise ValueError(
                    "If `architecture` is not provided," \
                    "`output_dim` must be specified."
                )
            self.architecture = nn.Linear(state_dim, output_dim)
    
    def encode(self, observation):
        return self.architecture.forward(observation)

    def get_output_dim(self):
        if hasattr(self.architecture, "out_features"):
            return self.architecture.out_features
        raise NotImplementedError(
            "Cannot infer output dim from custom architecture."
        )
