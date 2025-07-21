from abc import ABC, abstractmethod
from typing import Union
from .base_encoder import BaseEncoder
import torch
import torch.nn as nn
import numpy as np

class GridEncoder(BaseEncoder, ABC):
    def __init__(self, observation_space, architecture: "BaseGridArchitecture"):
        super().__init__()
        
        self.architecture = architecture

        obs_shape = observation_space.shape
        if len(obs_shape) == 3: # (H, W, C) or (C, H, W)
            if obs_shape[0] < obs_shape[2] and obs_shape[0] < obs_shape[1]:
                # (C, H, W)
                self.in_channels = obs_shape[0]
                self.height = obs_shape[1]
                self.width = obs_shape[2]
                self._needs_permute_to_nchw = False
            else:
                # (H, W, C)
                self.height = obs_shape[0]
                self.width = obs_shape[1]
                self.in_channels = obs_shape[2]
                self._needs_permute_to_nchw = True
        else: # (H, W) gray scale
            self.in_channels = 1
            self.height = obs_shape[0]
            self.width = obs_shape[1]
            self._needs_permute_to_nchw = False
        
        self._output_dim = self.architecture.get_output_dim(
            (1, self.in_channels, self.height, self.width)
        )

    def _format_observation(self, observation: np.ndarray) -> torch.Tensor:
        obs_tensor = torch.from_numpy(observation).float()

        if obs_tensor.dim() == len(self.observation_space.shape):
            obs_tensor = obs_tensor.unsqueeze(0)
        
        if self.in_channels == 1 and obs_tensor.dim() == 3:
            obs_tensor = obs_tensor.unsqueeze(1)

        elif self._needs_permute_to_nchw:
            obs_tensor = obs_tensor.permute(0, 3, 1, 2)

        if self.observation_space.high.max() > 1.0:
            obs_tensor = obs_tensor / 255.0

        return obs_tensor

    def encode(self, observation: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(observation, torch.Tensor):
            if observation.dim() == len(self.observation_space.shape):
                x = observation.unsqueeze(0)
            else:
                x = observation
        else:
            x = self._format_observation(observation)

        encoded_features = self.architecture(x)

        if isinstance(observation, np.ndarray) and observation.ndim == len(self.observation_space.shape):
             return encoded_features.squeeze(0)
        return encoded_features

    def get_output_dim(self):
        return self._output_dim

class BaseGridArchitecture(nn.Module, ABC):
    """
    Abstract class for handling different architecture for grid inputs.
    All architecture implementation, such as CNN, ResNet, or ViT, should inherit
    this class.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deal with the input tensor.
        (N, C, H, W) is the expected shape.
        """
        pass

    @abstractmethod
    def get_output_dim(self, input_nchw_shape: tuple) -> int:
        """
        With (N, C, H, W) being the given input shape,
        get the corresponding output shape.
        """
        pass

class CNNGridEncoder(BaseGridArchitecture):
    def __init__(self, in_channels: int, height: int, width: int):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )
        self._ouput_dim_val = self.get_output_dim(
            (1, in_channels, height, width)
        )

    def forward(self, x: torch.Tensor):
        features = self.conv_layers(x)
        return features.reshape(features.size(0), -1)
    
    def get_output_dim(self, input_nchw_shape: tuple) -> int:
        dummy_input = torch.zeros(input_nchw_shape)
        with torch.no_grad():
            dummy_output = self.conv_layers(dummy_input)
            return np.prod(dummy_output.shape[1:]).item()
