import cv2
import torch
import numpy as np
from typing import Dict, List, Optional

from .models.implementations import YOLO, SAM, BLIP, DAN, CLIP
from .database_wrapper import DatabaseWrapper
from .shepherd_config import ShepherdConfig

class Shepherd:
    def __init__(self, config : ShepherdConfig|None = None, database: DatabaseWrapper|None = None):
        """
        Initialize the Shepherd class with all required models and configurations.
        
        Args:
            config (ShepherdConfig): Configuration containing model paths and parameters
            database (DatabaseWrapper): Database wrapper instance for storing object data
        """
        if config is None:
            config = ShepherdConfig()
            
        if database is None:
            database = DatabaseWrapper(camera_utils=config.camera)

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
        
        # Initialize query embedding
        self.query_embedding = None
        if self.config.default_query:
            self.update_query(self.config.default_query)
        
    def update_query(self, query_text: str):
        """Update the query and compute its embedding."""
        self.config.default_query = query_text
        if query_text:
            # Convert query embedding to numpy array
            query_embedding = self.embedder.encode_text(query_text)
            if isinstance(query_embedding, str):
                query_embedding = None
            else:
                # Ensure it's a numpy array and normalized
                query_embedding = np.array(query_embedding)
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
        else:
            query_embedding = None
        
        # Update query embedding in database
        self.database.update_query(query_embedding)
        self.query_embedding = query_embedding
        
    def process_frame(self, frame: np.ndarray, depth_frame: Optional[np.ndarray] = None, camera_pose: Optional[Dict] = None):
        """Process a single frame through the vision pipeline."""
        # Run detection with lower confidence threshold for more detections
        detections = self._process_detections(frame)
        
        # Get segmentation masks for detections
        masks = self._process_segments(frame, detections)
        
        # If no depth frame provided, estimate it using DAN
        if depth_frame is None and self.depth_estimator is not None:
            depth_frame = self.depth_estimator.estimate_depth(frame)
            # Normalize depth values to a reasonable range (e.g. 0.1 to 10 meters)
            if depth_frame is not None:
                depth_frame = np.clip(depth_frame, 0.1, 10.0)
        
        # Process each detection
        results = []
        for detection, mask in zip(detections, masks):
            # Get bounding box coordinates from mask
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            
            # Ensure minimum size for processing
            if w < 10 or h < 10:
                continue

            # Extract region using bounding box with padding
            pad = 10  # Add padding pixels
            y1 = max(0, y-pad)
            y2 = min(frame.shape[0], y+h+pad)
            x1 = max(0, x-pad)
            x2 = min(frame.shape[1], x+w+pad)
            masked_region = frame[y1:y2, x1:x2]
            
            # Get caption and embedding
            if self.config.use_caption:
                caption = self._process_captions(masked_region)
            else:
                caption = None
            
            embedding = self.embedder.encode_image(masked_region)
            
            # Get depth information and create point cloud
            depth_info = None
            point_cloud = None
            if depth_frame is not None:
                # Use the full mask for depth processing
                depth_info = self._get_depth_info(mask, depth_frame)
                point_cloud = self._create_point_cloud(mask, depth_frame)
            
            # Only process if we have valid point cloud data
            if point_cloud is not None and len(point_cloud) > 10:  # Require minimum points
                # Create metadata
                metadata = {
                    'caption': caption,
                    'class_id': detection.get('class_id', 0),
                    'confidence': detection['confidence'],
                    'depth_info': depth_info,
                }
                
                # Store in database and get object ID
                object_id = self.database.store_object(
                    embedding=embedding,
                    metadata=metadata,
                    point_cloud=point_cloud,
                    camera_pose=camera_pose,
                    mask=mask
                )
                
                # Get similarity from metadata if it exists
                similarity = metadata.get('query_similarity')
                
                results.append({
                    'detection': detection,
                    'mask': mask,
                    'caption': caption,
                    'embedding': embedding,
                    'depth_info': depth_info,
                    'object_id': object_id,
                    'similarity': similarity
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
        
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0
            
            # Convert to numpy arrays if needed
            if isinstance(embedding1, torch.Tensor):
                embedding1 = embedding1.detach().cpu().numpy()
            if isinstance(embedding2, torch.Tensor):
                embedding2 = embedding2.detach().cpu().numpy()
            
            embedding1 = np.array(embedding1)
            embedding2 = np.array(embedding2)
            
            # Ensure embeddings are normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            embedding1 = embedding1 / norm1
            embedding2 = embedding2 / norm2
            
            return float(np.dot(embedding1, embedding2))
        except Exception as e:
            print(f"Error in similarity computation: {e}")
            return 0.0
        
        