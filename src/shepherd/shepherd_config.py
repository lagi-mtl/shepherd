from pathlib import Path
from typing import Dict, Any, Optional
import os
import torch
import numpy as np
from .utils.camera import CameraUtils

class ShepherdConfig:
    def __init__(self, model_dir: Optional[str] = None, 
                 camera_height: float = 0.4, 
                 camera_pitch: float = -1.57):  # -90 degrees in radians
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
            'blip': str(self.model_dir / 'blip-image-captioning-base'),
            'clip': str(self.model_dir / 'ViT-B-32.pt'),
            'dan': str(self.model_dir / 'dan.pt')
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
        
        # Initialize camera utils
        self.camera = CameraUtils(
            width=1344,
            height=376,
            fov=1.88,
            camera_height=camera_height,
            camera_pitch=camera_pitch
        )
        
        # Print camera parameters
        print(f"\nCamera parameters:")
        print(f"Image size: {self.camera.width}x{self.camera.height}")
        print(f"FOV: {np.degrees(self.camera.fov):.1f} degrees")
        print(f"Focal length: fx={self.camera.fx:.1f}, fy={self.camera.fy:.1f}")
        print(f"Principal point: cx={self.camera.cx:.1f}, cy={self.camera.cy:.1f}")
        print(f"Camera height: {self.camera.camera_height:.2f}m")
        print(f"Camera pitch: {np.degrees(self.camera.camera_pitch):.1f} degrees")
        
        # Add default query
        self.default_query = "nice place to sit"
        
        # Print model paths for debugging
        print("Model paths:")
        for model, path in self.model_paths.items():
            print(f"  {model}: {path}")
        print(f"Default query: {self.default_query}")
    
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

