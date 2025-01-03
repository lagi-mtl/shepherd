from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Any, Dict


class BaseModel(ABC):
    def __init__(self, model_path: str, device: str):
        """
        Initialize base model.

        Args:
            model_path (str): Path to model weights
            device (str): Device to run model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self.load_model()

    @abstractmethod
    def load_model(self):
        """Load model from path."""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess input data."""
        pass

    @abstractmethod
    def postprocess(self, output: Any) -> Dict:
        """Postprocess model output."""
        pass
