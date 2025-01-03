"""
Captioning model for generating captions from images.
"""

from abc import abstractmethod

import numpy as np

from .base_model import BaseModel


class CaptioningModel(BaseModel):
    """
    Base class for captioning models.
    """

    @abstractmethod
    def generate_caption(self, image: np.ndarray) -> str:
        """
        Generate caption for image.
        Returns string caption.
        """
