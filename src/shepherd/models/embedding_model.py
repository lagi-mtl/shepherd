"""
Embedding model.
"""

from abc import abstractmethod

import numpy as np

from .base_model import BaseModel


class EmbeddingModel(BaseModel):
    """
    Base class for embedding models.
    """

    @abstractmethod
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Encode image to embedding vector.
        """

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector.
        """
