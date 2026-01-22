import torch
from torch import nn
from .base_encoder import BaseEncoder
from typing import List, Union

class CompositeEncoder(BaseEncoder):
    """
    Composes multiple encoders by concatenating their feature outputs.
    """
    def __init__(self, encoders: List[BaseEncoder]):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        # Sum of all injected encoders' feature dimensions
        self._feature_dim = sum(e.feature_dim for e in encoders)

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]):
        # Handle both single tensor (for simple cases) or list of tensors
        if not isinstance(x, list):
            x = [x]
            
        features = [enc(data) for enc, data in zip(self.encoders, x)]
        return torch.cat(features, dim=-1)

    @property
    def feature_dim(self):
        return self._feature_dim
