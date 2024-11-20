import numpy as np
from typing import Dict, Tuple

class CameraUtils:
    def __init__(self, width: int = 1344, height: int = 376, fov: float = 1.88, 
                 camera_height: float = 0.4, camera_pitch: float = -1.57):  # -90 degrees in radians
        """
        Initialize camera parameters.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            fov: Horizontal field of view in radians
            camera_height: Camera height from ground in meters
            camera_pitch: Camera pitch angle in radians (negative is looking down)
        """
        self.width = width
        self.height = height
        self.fov = fov
        self.camera_height = camera_height
        self.camera_pitch = camera_pitch
        
        # Calculate focal length from FOV and image width
        self.fx = (self.width / 2) / np.tan(self.fov / 2)
        self.fy = self.fx  # Assuming square pixels
        self.cx = self.width / 2
        self.cy = self.height / 2
        
        # Create fixed camera rotation matrices
        self.Rx = np.array([  # Pitch rotation (around X)
            [1, 0, 0],
            [0, np.cos(camera_pitch), -np.sin(camera_pitch)],
            [0, np.sin(camera_pitch), np.cos(camera_pitch)]
        ])
        
        self.Rz = np.array([  # -90° around Z (yaw)
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        
        self.Rz_180 = np.array([  # 180° rotation around Z
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        
        # Combined camera rotation matrix
        self.camera_rotation = self.Rz_180 @ self.Rz @ self.Rx
        
    def transform_point_cloud(self, points: np.ndarray, camera_pose: Dict) -> np.ndarray:
        """Transform entire point cloud from camera to world coordinates."""
        # Apply camera rotation to all points
        rotated = (self.camera_rotation @ points.T).T
        
        # Add camera offset to all points
        camera_offset = np.array([0.1, 0, self.camera_height])
        transformed = rotated + camera_offset
        
        # Apply camera pose transformation if provided
        if camera_pose:
            qx, qy, qz, qw = camera_pose['qx'], camera_pose['qy'], camera_pose['qz'], camera_pose['qw']
            R = np.array([
                [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
            ])
            t = np.array([camera_pose['x'], camera_pose['y'], camera_pose['z']])
            transformed = (R @ transformed.T).T + t
            
        return transformed 