from typing import Dict, Tuple

import numpy as np


class CameraUtils:
    def __init__(
        self,
        width: int = 1344,
        height: int = 376,
        fov: float = 1.88,
        camera_height: float = 0.4,
        camera_pitch: float = 0.0,
        camera_yaw: float = 0.0,
        camera_roll: float = 0.0,
    ):
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
        self.Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(self.camera_pitch), -np.sin(self.camera_pitch)],
                [0, np.sin(self.camera_pitch), np.cos(self.camera_pitch)],
            ]
        )

        # Yaw rotation (around Y)
        self.Ry = np.array(
            [
                [np.cos(self.camera_yaw), 0, np.sin(self.camera_yaw)],
                [0, 1, 0],
                [-np.sin(self.camera_yaw), 0, np.cos(self.camera_yaw)],
            ]
        )

        # Roll rotation (around Z)
        self.Rz = np.array(
            [
                [np.cos(self.camera_roll), -np.sin(self.camera_roll), 0],
                [np.sin(self.camera_roll), np.cos(self.camera_roll), 0],
                [0, 0, 1],
            ]
        )

        # Combined camera rotation matrix (apply in order: roll, pitch, yaw)
        self.camera_rotation = self.Ry @ self.Rx @ self.Rz

    def unproject_points(self, points: np.ndarray) -> np.ndarray:
        """
        Unproject points from image coordinates to camera coordinates.
        Points are expected to be in [X, Y, Z] format where:
        - X and Y are image coordinates (relative to principal point)
        - Z is depth
        """
        # Get focal lengths and principal point
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy
        
        # Unproject
        # Convert from image coordinates to camera coordinates
        X = (points[:, 0] - cx) * points[:, 2] / fx
        Y = (points[:, 1] - cy) * points[:, 2] / fy
        Z = points[:, 2]  # depth remains the same
        
        return np.stack([X, Y, Z], axis=1)

    def transform_point_cloud(
        self, points: np.ndarray, camera_pose: Dict
    ) -> np.ndarray:
        """Transform points from camera to world coordinates using Habitat convention."""
        if not camera_pose:
            return points
        
        print("\n=== Transform Debug ===")
        print(f"Input points:\n{points[:3]}")
        
        # In Habitat camera space:
        # +X right, +Y down, +Z forward
        # In Habitat world space:
        # +X right, +Y up, +Z forward
        
        # 1. Convert from camera coords (Y-down) to world coords (Y-up)
        points_world = points.copy()
        points_world[:, 1] *= -1  # Flip Y
        print(f"\nAfter Y flip:\n{points_world[:3]}")
        
        # 2. Add homogeneous coordinate
        points_h = np.hstack([points_world, np.ones((len(points_world), 1))])
        
        # 3. Build world transform
        qx, qy, qz, qw = (
            camera_pose["qx"],
            camera_pose["qy"],
            camera_pose["qz"],
            camera_pose["qw"],
        )
        
        # Convert quaternion to rotation matrix (Habitat convention)
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz,  2*qx*qy - 2*qz*qw,    2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,      1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,    1 - 2*qx*qx - 2*qy*qy]
        ])
        
        # Build transform matrix
        T_world_camera = np.eye(4)
        T_world_camera[:3, :3] = R
        T_world_camera[:3, 3] = [camera_pose["x"], camera_pose["y"], camera_pose["z"]]
        
        print(f"\nCamera pose: {camera_pose}")
        print(f"Transform matrix:\n{T_world_camera}")
        
        # 4. Transform to world coordinates
        transformed = (T_world_camera @ points_h.T).T
        print(f"\nFinal points:\n{transformed[:3]}")
        
        return transformed[:, :3]

    def save_debug_cloud(self, points: np.ndarray, filename: str, color: Tuple[float, float, float]):
        """Save a point cloud with specific color for debugging."""
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Set all points to the specified color
        colors = np.tile(color, (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(filename, pcd)

    def update_rotation_matrices(self):
        """Update rotation matrices based on current angles."""
        # Note: These rotations are in world space (Y up)
        
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

        # Combined rotation: Yaw * Pitch * Roll
        self.camera_rotation = self.Ry @ self.Rx @ self.Rz
    
    def debug_transform(self, points: np.ndarray, camera_pose: Dict):
        """Debug coordinate transforms by printing/visualizing intermediate steps."""
        print("\nDEBUG COORDINATE TRANSFORM:")
        print(f"Input points shape: {points.shape}")
        print(f"Sample input points:\n{points[:3]}")  # Show first 3 points
        
        # After camera rotation
        rotated = (self.camera_rotation @ points.T).T
        print(f"\nAfter camera rotation:\n{rotated[:3]}")
        
        # After camera offset
        camera_offset = np.array([0.1, 0, self.camera_height])
        transformed = rotated + camera_offset
        print(f"\nAfter camera offset:\n{transformed[:3]}")
        print(f"Camera offset: {camera_offset}")
        
        # After pose transform
        if camera_pose:
            print(f"\nCamera pose: {camera_pose}")
            qx, qy, qz, qw = camera_pose["qx"], camera_pose["qy"], camera_pose["qz"], camera_pose["qw"]
            R = np.array([
                [1 - 2*qy*qy - 2*qz*qz,  2*qx*qy - 2*qz*qw,    2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw,      1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,    1 - 2*qx*qx - 2*qy*qy]
            ])
            print(f"\nRotation matrix:\n{R}")
            
            t = np.array([camera_pose["x"], camera_pose["y"], camera_pose["z"]])
            final = (R @ transformed.T).T + t
            print(f"\nFinal transformed points:\n{final[:3]}")
