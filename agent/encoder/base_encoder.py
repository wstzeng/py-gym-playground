from abc import ABC, abstractmethod
import torch.nn as nn

class BaseEncoder(nn.Module, ABC):
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        pass
