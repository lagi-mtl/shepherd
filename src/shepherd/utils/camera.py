from typing import Dict, Literal, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


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
        coordinate_frame: Literal["ros", "habitat"] = "ros",
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
            coordinate_frame: Coordinate frame convention ("ros" or "habitat")
        """
        self.width = width
        self.height = height
        self.fov = fov
        self.camera_height = camera_height
        self.coordinate_frame = coordinate_frame

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
        """Update rotation matrices based on current angles and coordinate frame."""
        if self.coordinate_frame == "habitat":
            # Habitat uses Y-up, -Z forward
            # First convert from camera frame (Z forward, -Y up) to Habitat frame
            base_rotation = Rotation.from_euler("xyz", [0, -90, 90], degrees=True)

            # Apply camera rotations in Habitat's frame
            camera_rotation = Rotation.from_euler(
                "yxz",
                [
                    -self.camera_yaw,  # Negated because Habitat's yaw is opposite
                    -self.camera_pitch,  # Negated to match Habitat's convention
                    self.camera_roll,
                ],
            )

            combined_rotation = camera_rotation * base_rotation  # Order matters here
            self.camera_rotation = combined_rotation.as_matrix()
        else:
            # ROS/Gazebo coordinate frame (Z-up, X forward)
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

    def transform_point_cloud(
        self, points: np.ndarray, camera_pose: Dict
    ) -> np.ndarray:
        """Transform entire point cloud from camera to world coordinates."""
        if len(points) == 0:
            return points

        if self.coordinate_frame == "habitat":
            # Convert points from camera frame to Habitat world frame
            points_fixed = points.copy()

            # 1. Convert from depth camera frame to Habitat camera frame
            # Depth camera: Z forward, -Y up, X right
            # Habitat: X right, Y up, -Z forward
            points_fixed[..., [0, 1, 2]] = points[..., [0, 1, 2]]  # Keep X as is
            points_fixed[..., 1] *= -1  # Flip Y (up)
            points_fixed[..., 2] *= -1  # Flip Z (forward)

            if camera_pose:
                # 2. Get rotation matrix from quaternion (raw Habitat quaternion)
                R = Rotation.from_quat(
                    [
                        camera_pose["qx"],
                        camera_pose["qy"],
                        camera_pose["qz"],
                        camera_pose["qw"],
                    ]
                ).as_matrix()

                # 3. Apply rotation
                points_fixed = (R @ points_fixed.T).T

                # 4. Apply translation
                t = np.array(
                    [
                        camera_pose["x"],
                        camera_pose["y"],
                        camera_pose["z"],
                    ]
                )
                points_fixed = points_fixed + t

            # 5. Add camera height offset
            camera_offset = np.array([0, self.camera_height, 0])
            transformed = points_fixed + camera_offset

            return transformed
        else:
            # Original ROS transformation code...
            rotated = (self.camera_rotation @ points.T).T
            camera_offset = np.array([0.1, 0, self.camera_height])
            transformed = rotated + camera_offset

            if camera_pose:
                qx, qy, qz, qw = (
                    camera_pose["qx"],
                    camera_pose["qy"],
                    camera_pose["qz"],
                    camera_pose["qw"],
                )

                R = np.array(
                    [
                        [
                            1 - 2 * qy * qy - 2 * qz * qz,
                            2 * qx * qy - 2 * qz * qw,
                            2 * qx * qz + 2 * qy * qw,
                        ],
                        [
                            2 * qx * qy + 2 * qz * qw,
                            1 - 2 * qx * qx - 2 * qz * qz,
                            2 * qy * qz - 2 * qx * qw,
                        ],
                        [
                            2 * qx * qz - 2 * qy * qw,
                            2 * qy * qz + 2 * qx * qw,
                            1 - 2 * qx * qx - 2 * qy * qy,
                        ],
                    ]
                )

                t = np.array([camera_pose["x"], camera_pose["y"], camera_pose["z"]])
                transformed = (R @ transformed.T).T + t

        return transformed
