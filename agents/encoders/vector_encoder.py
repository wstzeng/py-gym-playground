import torch
import torch.nn as nn
from .base_encoder import BaseEncoder

class VectorEncoder(BaseEncoder):
    def __init__(self, state_dim):
        super().__init__(state_dim)
    

