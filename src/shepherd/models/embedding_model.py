from abc import abstractmethod
import numpy as np
from .base_model import BaseModel


class EmbeddingModel(BaseModel):
    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)

    @abstractmethod
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Encode image to embedding vector.
        """
        pass

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector.
        """
        pass
