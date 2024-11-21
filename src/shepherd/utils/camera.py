import numpy as np
from typing import Dict, Tuple

class CameraUtils:
    def __init__(self, width: int = 1344, height: int = 376, fov: float = 1.88, 
                 camera_height: float = 0.4, camera_pitch: float = 0.0,
                 camera_yaw: float = 0.0, camera_roll: float = 0.0):
        """
        Initialize camera parameters.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            fov: Horizontal field of view in radians
            camera_height: Camera height from ground in meters
            camera_pitch: Camera pitch angle in radians
            camera_yaw: Camera yaw angle in radians
            camera_roll: Camera roll angle in radians
        """
        self.width = width
        self.height = height
        self.fov = fov
        self.camera_height = camera_height
        
        # Store rotation angles
        self.camera_pitch = camera_pitch
        self.camera_yaw = camera_yaw
        self.camera_roll = camera_roll
        
        # Calculate focal length from FOV and image width
        self.fx = (self.width / 2) / np.tan(self.fov / 2)
        self.fy = self.fx  # Assuming square pixels
        self.cx = self.width / 2
        self.cy = self.height / 2
        
        # Create rotation matrices
        self.update_rotation_matrices()
        
    def update_rotation_matrices(self):
        """Update rotation matrices based on current angles."""
        # Pitch rotation (around X)
        self.Rx = np.array([
            [1, 0, 0],
            [0, np.cos(self.camera_pitch), -np.sin(self.camera_pitch)],
            [0, np.sin(self.camera_pitch), np.cos(self.camera_pitch)]
        ])
        
        # Yaw rotation (around Y)
        self.Ry = np.array([
            [np.cos(self.camera_yaw), 0, np.sin(self.camera_yaw)],
            [0, 1, 0],
            [-np.sin(self.camera_yaw), 0, np.cos(self.camera_yaw)]
        ])
        
        # Roll rotation (around Z)
        self.Rz = np.array([
            [np.cos(self.camera_roll), -np.sin(self.camera_roll), 0],
            [np.sin(self.camera_roll), np.cos(self.camera_roll), 0],
            [0, 0, 1]
        ])
        
        # Combined camera rotation matrix (apply in order: roll, pitch, yaw)
        self.camera_rotation = self.Ry @ self.Rx @ self.Rz
        
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