"""
Segmentation model.
"""

from abc import abstractmethod
from typing import Dict, List

import numpy as np

from .base_model import BaseModel


class SegmentationModel(BaseModel):
    """
    Base class for segmentation models.
    """

    @abstractmethod
    def segment(self, image: np.ndarray, detections: List[Dict]) -> List[np.ndarray]:
        """
        Segment objects in image.
        Returns list of segmentation masks.
        """
