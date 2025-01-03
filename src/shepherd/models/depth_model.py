"""
Depth model for estimating depth from 2D images.
"""

from abc import abstractmethod

import numpy as np

from .base_model import BaseModel


class DepthModel(BaseModel):
    """
    Base class for depth estimation models.
    """

    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)

    @abstractmethod
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from image.
        Returns depth map as numpy array.
        """
        pass
