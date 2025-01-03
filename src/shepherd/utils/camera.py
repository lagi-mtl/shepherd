from typing import Dict, Literal, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


class CameraUtils:
    # Coordinate system definitions
    COORDINATE_SYSTEMS = {
        "ros": {
            "up_axis": "z",
            "forward_axis": "x",
            "camera_offset": np.array([0.1, 0, 0]),  # Offset in camera frame
        },
        "habitat": {
            "up_axis": "y",
            "forward_axis": "-z",
            "camera_offset": np.array([0, 0, 0]),  # No additional offset needed
        },
    }

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
        """Initialize camera parameters."""
        self.width = width
        self.height = height
        self.fov = fov
        self.camera_height = camera_height
        self.coordinate_frame = coordinate_frame

        # Camera intrinsics
        self.fx = (self.width / 2) / np.tan(self.fov / 2)
        self.fy = self.fx  # Assuming square pixels
        self.cx = self.width / 2
        self.cy = self.height / 2

        # Camera extrinsics
        self.set_camera_pose(camera_pitch, camera_yaw, camera_roll)

    def set_camera_pose(self, pitch: float, yaw: float, roll: float):
        """Update camera pose angles."""
        self.camera_pitch = pitch
        self.camera_yaw = yaw
        self.camera_roll = roll

        # Create camera rotation from euler angles
        if self.coordinate_frame == "habitat":
            # Habitat uses different angle conventions
            self.camera_rotation = Rotation.from_euler(
                "yxz",  # Habitat's rotation order
                [-yaw, -pitch, roll],  # Negated to match Habitat's convention
            ).as_matrix()
        else:
            # ROS/Gazebo convention
            self.camera_rotation = Rotation.from_euler(
                "xyz", [roll, pitch, yaw]
            ).as_matrix()

    def transform_point_cloud(
        self, points: np.ndarray, camera_pose: Dict
    ) -> np.ndarray:
        """Transform point cloud from camera to world coordinates."""
        if len(points) == 0:
            return points

        # Get coordinate system configuration
        coord_sys = self.COORDINATE_SYSTEMS[self.coordinate_frame]

        # 1. Convert points from camera frame to world-aligned frame
        points_fixed = points.copy()

        if self.coordinate_frame == "habitat":
            # Convert from camera (Z forward, -Y up) to Habitat (X right, Y up, -Z forward)
            points_fixed[..., 1] *= -1  # Flip Y (up)
            points_fixed[..., 2] *= -1  # Flip Z (forward)

        # 2. Apply camera pose transformation
        if camera_pose:
            # Get world rotation from quaternion
            R = Rotation.from_quat(
                [
                    camera_pose["qx"],
                    camera_pose["qy"],
                    camera_pose["qz"],
                    camera_pose["qw"],
                ]
            ).as_matrix()

            # Apply rotation and translation
            points_fixed = (R @ points_fixed.T).T
            t = np.array([camera_pose["x"], camera_pose["y"], camera_pose["z"]])
            points_fixed = points_fixed + t

        # 3. Add camera height and system-specific offset
        height_offset = np.zeros(3)
        if coord_sys["up_axis"] == "y":
            height_offset[1] = self.camera_height
        else:  # "z"
            height_offset[2] = self.camera_height

        transformed = points_fixed + height_offset + coord_sys["camera_offset"]

        return transformed
