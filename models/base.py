from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load(self, weights_path: str):
        """Load model weights from file"""
        pass

    @abstractmethod
    def preprocess(self, x):
        """Preprocess input data before forward pass"""
        pass

    @abstractmethod
    def forward(self, x):
        """Forward pass of the model"""
        pass
