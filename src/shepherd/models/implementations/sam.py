import torch
from ultralytics.models.fastsam import FastSAMPredictor
import numpy as np
from typing import List, Dict
from ..segmentation_model import SegmentationModel
import cv2


class SAM(SegmentationModel):
    def __init__(
        self,
        model_path: str,
        device: str,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
    ):
        super().__init__(model_path, device)
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.original_width = None
        self.original_height = None

    def load_model(self):
        """Load FastSAM model."""
        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            imgsz=1024,
            model=self.model_path,
            save=False,
        )
        self.predictor = FastSAMPredictor(overrides=overrides)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for FastSAM."""
        self.original_height, self.original_width = image.shape[:2]
        return cv2.resize(image, (1024, 1024))

    def segment(self, image: np.ndarray, detections: List[Dict]) -> List[np.ndarray]:
        """Generate segmentation masks using FastSAM with YOLO detections."""
        image = self.preprocess(image)
        results = self.predictor(image)

        # Get masks for each detection
        masks = []
        for detection in detections:
            bbox = detection["bbox"]
            # Scale bbox to 1024x1024
            scale_x = 1024 / self.original_width
            scale_y = 1024 / self.original_height
            scaled_bbox = [
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y,
            ]
            prompt_results = self.predictor.prompt(results, scaled_bbox)
            if prompt_results and len(prompt_results) > 0:
                mask = prompt_results[0].masks.data[0].cpu().numpy()
                masks.append(mask)

        return self.postprocess(masks)

    def postprocess(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """Resize masks back to original size."""
        processed_masks = []
        for mask in masks:
            resized_mask = cv2.resize(
                mask.astype(float), (self.original_width, self.original_height)
            )
            processed_masks.append(resized_mask > 0.5)
        return processed_masks
