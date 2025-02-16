import os
import time
from typing import Dict, Tuple

import cv2
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation

from src.shepherd import Shepherd, ShepherdConfig
from src.shepherd.utils.camera import CameraUtils


class PyBulletShepherd:
    def __init__(self, render_width=512, render_height=512):
        # Initialize PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Rendering parameters
        self.render_width = render_width
        self.render_height = render_height
        
        # Camera parameters
        self.camera_position = np.array([0.0, -2.0, 1])
        self.camera_yaw = 0
        self.camera_pitch = 0
        self.move_speed = 0.1
        self.turn_speed = 5
        
        # Initialize Shepherd
        self.config = ShepherdConfig(
            camera_height=1,  # Camera height from ground in meters
            camera_pitch=np.deg2rad(0),  # Initial pitch in radians
        )
        
        # Update camera parameters
        self.config.camera = CameraUtils(
            width=render_width,
            height=render_height,
            fov=np.deg2rad(60),  # 60 degrees FOV
            camera_height=1,
            camera_pitch=np.deg2rad(0),
            camera_yaw=0,
            camera_roll=0,
        )
        
        # Initialize Shepherd
        self.shepherd = Shepherd(config=self.config)
        
        # Load scene
        self._load_scene()
        
        # Frame processing control
        self.frame_skip = 2
        self.frame_count = 0
        
        # Setup visualization
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        cv2.namedWindow("Shepherd View", cv2.WINDOW_NORMAL)
        
        # Create output directory
        self.output_dir = os.path.join(os.getcwd(), "demo/pybullet_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Color mapping for visualization
        self.color_map = {}
        self.next_color_idx = 0
        self.color_palette = [
            (255, 0, 0),   # Blue
            (0, 255, 0),   # Green
            (0, 0, 255),   # Red
            (255, 255, 0), # Cyan
            (255, 0, 255), # Magenta
            (0, 255, 255), # Yellow
        ]

    def _load_scene(self):
        """Load the PyBullet scene."""
        p.loadURDF("plane.urdf")
        self.table = p.loadURDF("table/table.urdf", [0, 0, 0])
        self.cube = p.loadURDF("cube_small.urdf", [0.0, 0, 0.8])
        self.duck = p.loadURDF("duck_vhacd.urdf", [0.3, 0, 0.8])
        self.sphere = p.loadURDF("sphere2red.urdf", [-0.3, 0, 0.8], globalScaling=0.3)

    def get_camera_view(self):
        """Get camera view and depth information."""
        # Compute view matrix
        yaw_rad = np.deg2rad(self.camera_yaw)
        pitch_rad = np.deg2rad(self.camera_pitch)
        
        # Forward vector (remains unchanged for the view matrix)
        forward = np.array([
            np.sin(yaw_rad) * np.cos(pitch_rad),
            np.cos(yaw_rad) * np.cos(pitch_rad),
            -np.sin(pitch_rad)
        ])
        
        target = self.camera_position + forward
        up = [0, 0, 1]  # z-up
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=target,
            cameraUpVector=up
        )
        
        # Projection matrix
        fov = 60
        aspect = self.render_width / self.render_height
        near = 0.1
        far = 10
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far
        )
        
        # Get camera image
        img_arr = p.getCameraImage(
            width=self.render_width,
            height=self.render_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Process RGB and depth
        rgba = np.reshape(img_arr[2], (self.render_height, self.render_width, 4))
        rgb = rgba[:, :, :3].astype(np.uint8)
        
        depth_buffer = np.reshape(img_arr[3], (self.render_height, self.render_width))
        depth = far * near / (far - (far - near) * depth_buffer)
        
        desired_forward = forward
        base_rot, _ = Rotation.align_vectors([desired_forward], [np.array([0, 1, 0])])
        adjustment = Rotation.from_euler('x', -np.pi/2, degrees=False)
        camera_rot = base_rot * adjustment
        quat = camera_rot.as_quat()
        
        camera_pose = {
            "x": float(self.camera_position[0]),
            "y": float(self.camera_position[1]),
            "z": float(self.camera_position[2]),
            "qx": float(quat[0]),
            "qy": float(quat[1]),
            "qz": float(quat[2]),
            "qw": float(quat[3]),
        }
        
        return rgb, depth, camera_pose

    def get_object_color(self, object_id: str, similarity: float = None) -> Tuple[int, int, int]:
        """Get color based on query similarity or assign distinct color if no query."""
        if similarity is not None:
            intensity = int(similarity * 255)
            return (0, 0, intensity)
        else:
            if object_id not in self.color_map:
                color = self.color_palette[self.next_color_idx % len(self.color_palette)]
                self.color_map[object_id] = color
                self.next_color_idx += 1
            return self.color_map[object_id]

    def save_point_cloud(self):
        """Save the current point cloud."""
        output_path = os.path.join(self.output_dir, "pybullet_point_cloud.ply")
        self.shepherd.database.save_point_cloud_ply(output_path)
        print(f"Saved point cloud to: {output_path}")

    def run(self):
        """Main run loop."""
        print("\nControls:")
        print("W/S - Move forward/backward")
        print("A/D - Move left/right")
        print("Q/E - Turn left/right")
        print("R/F - Look up/down")
        print("K - Enter query")
        print("P - Save point cloud")
        print("ESC - Exit")
        
        try:
            while True:
                # Get camera view
                rgb, depth, camera_pose = self.get_camera_view()
                
                # Process frame with Shepherd
                self.frame_count += 1
                if self.frame_count % self.frame_skip == 0:
                    print(camera_pose)
                    

                    results = self.shepherd.process_frame(rgb, depth, camera_pose)
                    
                    # Create visualization
                    viz_frame = rgb.copy()
                    
                    # Draw detections and masks
                    for result in results:
                        bbox = result["detection"]["bbox"]
                        mask = result["mask"]
                        object_id = result.get("object_id")
                        similarity = result.get("similarity")
                        
                        # Get color
                        color = self.get_object_color(
                            object_id, 
                            similarity if self.shepherd.config.default_query else None
                        )
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw mask
                        mask_overlay = viz_frame.copy()
                        mask_overlay[mask] = color
                        viz_frame = cv2.addWeighted(viz_frame, 0.7, mask_overlay, 0.3, 0)
                        
                        # Add text
                        text = f"ID: {object_id}"
                        if similarity is not None:
                            text += f" ({similarity:.2f})"
                        cv2.putText(viz_frame, text, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add query information
                    query_text = f"Query: {self.shepherd.config.default_query}"
                    cv2.putText(viz_frame, query_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Show frame
                    cv2.imshow("Shepherd View", viz_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('w'):
                    self.camera_position[0] += self.move_speed * np.sin(np.deg2rad(self.camera_yaw))
                    self.camera_position[1] += self.move_speed * np.cos(np.deg2rad(self.camera_yaw))
                elif key == ord('s'):
                    self.camera_position[0] -= self.move_speed * np.sin(np.deg2rad(self.camera_yaw))
                    self.camera_position[1] -= self.move_speed * np.cos(np.deg2rad(self.camera_yaw))
                elif key == ord('a'):
                    self.camera_position[0] += self.move_speed * np.sin(np.deg2rad(self.camera_yaw - 90))
                    self.camera_position[1] += self.move_speed * np.cos(np.deg2rad(self.camera_yaw - 90))
                elif key == ord('d'):
                    self.camera_position[0] += self.move_speed * np.sin(np.deg2rad(self.camera_yaw + 90))
                    self.camera_position[1] += self.move_speed * np.cos(np.deg2rad(self.camera_yaw + 90))
                elif key == ord('q'):
                    self.camera_yaw -= self.turn_speed
                elif key == ord('e'):
                    self.camera_yaw += self.turn_speed
                elif key == ord('r'):
                    self.camera_pitch = min(self.camera_pitch + self.turn_speed, 89)
                elif key == ord('f'):
                    self.camera_pitch = max(self.camera_pitch - self.turn_speed, -89)
                elif key == ord('k'):
                    query = input("\nEnter query: ")
                    self.shepherd.update_query(query)
                elif key == ord('p'):
                    self.save_point_cloud()
                elif key == 27:  # ESC
                    break
                
                # Step simulation
                p.stepSimulation()
                time.sleep(0.01)
                
        finally:
            cv2.destroyAllWindows()
            p.disconnect()

def main():
    shepherd = PyBulletShepherd()
    shepherd.run()

if __name__ == "__main__":
    main() 