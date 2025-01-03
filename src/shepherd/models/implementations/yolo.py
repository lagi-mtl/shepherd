"""
YOLO model implementation.
"""

from typing import Any, Dict, List

import numpy as np
import torch
from ultralytics import YOLO as UltralyticsYOLO

from ..detection_model import DetectionModel


class YOLO(DetectionModel):
    """
    YOLO model implementation.
    """

    def load_model(self):
        """Load YOLO model."""
        self.model = UltralyticsYOLO(self.model_path)
        self.model.to(self.device)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """YOLO handles preprocessing internally."""
        return image

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run detection on image."""
        results = self.model(
            image, conf=self.confidence_threshold, iou=self.nms_threshold
        )
        return self.postprocess(results)

    def postprocess(self, output: Any) -> List[Dict]:
        """Convert YOLO output to list of detections."""
        detections = []
        for result in output:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "bbox": box.xyxy[0].cpu().numpy(),
                    "confidence": box.conf.item(),
                    "class_id": box.cls.item(),
                }
                detections.append(detection)
        return detections
