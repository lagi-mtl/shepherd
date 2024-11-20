import cv2
import torch
import numpy as np
from typing import Dict, List, Optional

from .models.implementations import YOLO, SAM, BLIP, DAN, CLIP
from .database_wrapper import DatabaseWrapper
from .shepherd_config import ShepherdConfig

class Shepherd:
    def __init__(self, config : ShepherdConfig = None, database: DatabaseWrapper = None):
        """
        Initialize the Shepherd class with all required models and configurations.
        
        Args:
            config (ShepherdConfig): Configuration containing model paths and parameters
            database (DatabaseWrapper): Database wrapper instance for storing object data
        """
        if config is None:
            config = ShepherdConfig()
            
        if database is None:
            database = DatabaseWrapper()

        self.config = config
        self.device = config.get('device', "cuda" if torch.cuda.is_available() else "cpu")
        self.database = database
        
        # Initialize models
        self.detector = YOLO(
            model_path=config.get('model_paths.yolo', "yolov8s-world.pt"),
            device=self.device,
            confidence_threshold=config.get('thresholds.detection', 0.4),
            nms_threshold=config.get('thresholds.nms', 0.45)
        )
        
        self.segmenter = SAM(
            model_path=config.get('model_paths.sam', "FastSAM-s.pt"),
            device=self.device,
            points_per_side=config.get('sam.points_per_side', 32),
            pred_iou_thresh=config.get('sam.pred_iou_thresh', 0.88)
        )
        
        self.captioner = BLIP(
            model_path=config.get('model_paths.blip', "Salesforce/blip-image-captioning-base"),
            device=self.device
        )
        
        self.embedder = CLIP(
            model_path=config.get('model_paths.clip', "ViT-B/32"),
            device=self.device
        )
        
        if config.get('use_depth', True):
            self.depth_estimator = DAN(
                model_path=config.get('model_paths.dan'),
                device=self.device
            )
        else:
            self.depth_estimator = None
            
        self._validate_models()
        
    def process_frame(self, frame: np.ndarray, depth_frame: Optional[np.ndarray] = None, camera_pose: Optional[Dict] = None):
        """Process a single frame through the vision pipeline."""
        # Run detection
        detections = self._process_detections(frame)
        
        # Get segmentation masks for detections
        masks = self._process_segments(frame, detections)
        
        # If no depth frame provided, estimate it using DAN
        if depth_frame is None and self.depth_estimator is not None:
            depth_frame = self.depth_estimator.estimate_depth(frame)
        
        # Process each detection
        results = []
        for detection, mask in zip(detections, masks):
            # Get bounding box coordinates from mask
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

            # Extract region using bounding box
            masked_region = frame[y:y+h, x:x+w]
            
            # Get caption and embedding
            caption = self._process_captions(masked_region)
            embedding = self.embedder.encode_image(masked_region)
            
            # Get depth information and create point cloud
            depth_info = None
            point_cloud = None

            depth_info = self._get_depth_info(mask, depth_frame)
            point_cloud = self._create_point_cloud(mask, depth_frame)
            
            # Store in database (point cloud transformation happens here)
            metadata = {
                'caption': caption,
                'class_id': detection.get('class_id', 0),
                'confidence': detection['confidence'],
                'depth_info': depth_info,
            }
            
            self.database.store_object(
                embedding=embedding,
                metadata=metadata,
                point_cloud=point_cloud,
                camera_pose=camera_pose
            )
            
            results.append({
                'detection': detection,
                'mask': mask,
                'caption': caption,
                'embedding': embedding,
                'depth_info': depth_info
            })
            
        return results

    def _validate_models(self):
        """Validate that all required models are properly initialized."""
        required_models = ['detector', 'segmenter', 'captioner', 'embedder']
        for model_name in required_models:
            if not hasattr(self, model_name) or getattr(self, model_name) is None:
                raise ValueError(f"Required model {model_name} is not properly initialized")
                
    def _process_detections(self, image: np.ndarray) -> List[Dict]:
        """Process image through detection model."""
        return self.detector.detect(image)
        
    def _process_segments(self, image: np.ndarray, detections: List[Dict]) -> List[np.ndarray]:
        """Process image through segmentation model."""
        return self.segmenter.segment(image, detections)
        
    def _process_captions(self, image: np.ndarray) -> str:
        """Generate captions for detected regions."""
        return self.captioner.generate_caption(image)
        
    def _get_masked_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract masked region from image."""
        # Get bounding box from mask
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0 or len(x_coords) == 0:
            return image
            
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Crop image to bounding box
        cropped = image[y_min:y_max, x_min:x_max]
        return cropped

    def _get_depth_info(self, mask: np.ndarray, depth_frame: np.ndarray) -> Dict:
        """Get depth statistics for masked region."""
        masked_depth = depth_frame[mask]
        if len(masked_depth) == 0:
            return None
            
        return {
            'min_depth': float(np.min(masked_depth)),
            'max_depth': float(np.max(masked_depth)),
            'mean_depth': float(np.mean(masked_depth)),
            'median_depth': float(np.median(masked_depth)),
            'depth_map': depth_frame
        }
        
    def _create_point_cloud(self, mask: np.ndarray, depth_frame: np.ndarray) -> np.ndarray:
        """Create point cloud from mask and depth frame using camera parameters."""
        # Get image dimensions
        height, width = depth_frame.shape
        
        # Create meshgrid of pixel coordinates
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        
        # Apply mask
        valid_points = mask > 0
        
        # Get valid coordinates and depths
        x = xx[valid_points]
        y = yy[valid_points]
        z = depth_frame[valid_points]
        
        # Filter out invalid depths
        valid_depths = z > 0.1
        x = x[valid_depths]
        y = y[valid_depths]
        z = z[valid_depths]
        
        # Convert to 3D coordinates using camera parameters
        X = (x - self.config.camera.cx) * z / self.config.camera.fx
        Y = (y - self.config.camera.cy) * z / self.config.camera.fy
        Z = z
        
        # Stack coordinates
        points = np.stack([X, Y, Z], axis=1)
        
        # Remove outliers
        if len(points) > 0:
            mean = np.mean(points, axis=0)
            std = np.std(points, axis=0)
            valid_points = np.all(np.abs(points - mean) <= 2 * std, axis=1)
            points = points[valid_points]
        
        return points
        
        