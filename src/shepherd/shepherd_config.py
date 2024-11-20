from pathlib import Path
from typing import Dict, Any, Optional
import os
import torch
import numpy as np
from .utils.camera import CameraUtils

class ShepherdConfig:
    def __init__(self, model_dir: Optional[str] = None, 
                 camera_height: float = 0.0, 
                 camera_pitch: float = 0.0):  # Default to 0 instead of -1.57
        """
        Initialize ShepherdConfig with model paths and parameters.
        
        Args:
            model_dir (Optional[str]): Path to directory containing model weights.
                                     If None, uses default 'model_weights' in project root.
        """
        # Get project root directory (2 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        
        # Set model weights directory
        if model_dir is None:
            self.model_dir = project_root / 'model_weights'
        else:
            self.model_dir = Path(model_dir)
            
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Default model paths - all models will be in the model_weights directory
        self.model_paths = {
            'yolo': str(self.model_dir / 'yolov8s-world.pt'),
            'sam': str(self.model_dir / 'FastSAM-s.pt'),
        }
        
        # Model-specific parameters
        self.thresholds = {
            'detection': 0.4,
            'nms': 0.45
        }
        
        self.sam_params = {
            'points_per_side': 32,
            'pred_iou_thresh': 0.88
        }
        
        # General settings
        self.use_depth = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize camera utils with default square frame
        self.camera = CameraUtils(
            width=1024,
            height=1024,
            fov=1.0,
            camera_height=camera_height,
            camera_pitch=camera_pitch
        )
        
        # Add default query
        self.default_query = "nice place to sit"
        print("default query :", self.default_query)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Example:
            config.get('model_paths.yolo')
            config.get('thresholds.detection')
        """
        try:
            current = self
            for part in key.split('.'):
                if hasattr(current, part):
                    current = getattr(current, part)
                elif isinstance(current, dict):
                    current = current[part]
                else:
                    return default
            return current
        except (AttributeError, KeyError):
            return default
            
    def __str__(self) -> str:
        """String representation of config for debugging."""
        return (
            f"ShepherdConfig:\n"
            f"  Model Directory: {self.model_dir}\n"
            f"  Model Paths: {self.model_paths}\n"
            f"  Thresholds: {self.thresholds}\n"
            f"  SAM Parameters: {self.sam_params}\n"
            f"  Use Depth: {self.use_depth}\n"
            f"  Device: {self.device}"
        )

