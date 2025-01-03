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

    @abstractmethod
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from image.
        Returns depth map as numpy array.
        """
        pass
