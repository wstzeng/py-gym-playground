from abc import ABC, abstractmethod
import torch.nn as nn

class BasePolicy(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass
