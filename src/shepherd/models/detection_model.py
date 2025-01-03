"""
Detection model for object detection.
"""

from abc import abstractmethod
from typing import Dict, List

import numpy as np

from .base_model import BaseModel


class DetectionModel(BaseModel):
    """
    Base class for object detection models.
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
    ):
        """
        Initialize detection model.

        Args:
            model_path (str): Path to model weights
            device (str): Device to run model on ('cuda' or 'cpu')
            confidence_threshold (float): Confidence threshold for detections
            nms_threshold (float): Non-maximum suppression threshold
        """
        super().__init__(model_path, device)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in image.

        Args:
            image (np.ndarray): Input image

        Returns:
            List[Dict]: List of detections, each containing bbox, confidence, and class_id
        """
