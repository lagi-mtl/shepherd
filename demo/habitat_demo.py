import os
import time
from datetime import datetime
from typing import Any, Dict, Tuple

import cv2
import gymnasium as gym
import habitat_sim
import numpy as np
import open3d as o3d
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from shepherd.shepherd import Shepherd
from shepherd.shepherd_config import ShepherdConfig
from shepherd.utils.camera import CameraUtils


class HabitatEnv(gym.Env):
    """RL environment wrapper for Habitat."""

    def __init__(self, scene_path: str, shepherd: Shepherd):
        super().__init__()

        # Store Shepherd instance
        self.shepherd = shepherd

        # Create simulator configuration
        self.cfg = self._make_sim_config(scene_path)

        # Initialize simulator
        self.sim = habitat_sim.Simulator(self.cfg)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(
            4
        )  # move_forward, turn_left, turn_right, do nothing

        # Observation space includes RGB image and depth image
        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
                "depth": spaces.Box(
                    low=0, high=np.inf, shape=(256, 256, 1), dtype=np.float32
                ),
            }
        )

        # Initialize agent
        self.agent = self.sim.initialize_agent(0)
        self._reset_agent()

        # Frame processing control
        self.last_frame_results = None
        self.last_frame_time = None
        self.frame_skip = 2  # Process every nth frame
        self.frame_count = 0

        # Create output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), "habitat_output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Add color map for visualization
        self.color_map = {}
        self.next_color_idx = 0
        self.color_palette = [
            (255, 0, 0),  # Blue
            (0, 255, 0),  # Green
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

    def _make_sim_config(self, scene_path: str) -> habitat_sim.Configuration:
        """Create Habitat simulator configuration."""
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0

        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        # RGB sensor
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.hfov = 90
        rgb_sensor_spec.resolution = [256, 256]
        rgb_sensor_spec.position = [0.0, 1.5, 0.0]  # Camera at agent head height
        rgb_sensor_spec.orientation = [0.0, 0.0, 0.0]  # No initial rotation

        # Depth sensor (matched to RGB sensor)
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.hfov = 90
        depth_sensor_spec.resolution = [256, 256]
        depth_sensor_spec.position = [0.0, 1.5, 0.0]
        depth_sensor_spec.orientation = [0.0, 0.0, 0.0]  # Match RGB sensor
        depth_sensor_spec.normalize_depth = False

        agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def _reset_agent(self):
        """Reset agent to initial position."""
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([0.0, 0.0, 0.0])  # At ground level
        # Initialize with identity rotation
        agent_state.rotation = np.quaternion(1.0, 0.0, 0.0, 0.0)
        self.agent.set_state(agent_state)

    def get_object_color(
        self, object_id: str, similarity: float = None
    ) -> Tuple[int, int, int]:
        """Get color based on query similarity or assign distinct color if no query."""
        if similarity is not None:
            intensity = int(similarity * 255)
            return (0, 0, intensity)
        else:
            if object_id not in self.color_map:
                color = self.color_palette[
                    self.next_color_idx % len(self.color_palette)
                ]
                self.color_map[object_id] = color
                self.next_color_idx += 1
            return self.color_map[object_id]

    def _get_camera_pose(self) -> Dict:
        """Get current agent pose in world coordinates with proper transformation."""
        agent_state = self.agent.get_state()
        position = agent_state.position

        # Just pass the raw Habitat quaternion to the camera utils
        habitat_rotation = agent_state.rotation

        return {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
            "qx": float(habitat_rotation.x),
            "qy": float(habitat_rotation.y),
            "qz": float(habitat_rotation.z),
            "qw": float(habitat_rotation.w),
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        """Execute action and return new observation."""
        # Map actions
        action_map = {
            0: "move_forward",
            1: "turn_left",
            2: "turn_right",
            3: "do_nothing",
        }

        # Execute action
        if action != 3:
            self.sim.step(action_map[action])

        # Get observation
        obs = self._get_observation()

        return obs, 0.0, False, False, {}

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation including vision processing results."""
        # Get raw sensor observations
        raw_obs = self.sim.get_sensor_observations()

        # Process RGB image
        rgb = raw_obs["color_sensor"]
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)

        # Process depth image
        depth = raw_obs["depth_sensor"]
        depth = depth.astype(np.float32)

        # Get agent pose
        agent_pose = self._get_camera_pose()

        # Process frame with Shepherd
        self.frame_count += 1
        if self.frame_count % self.frame_skip == 0:
            print("\n=== Processing New Frame ===")
            print(
                f"Agent Position: ({agent_pose['x']:.2f}, {agent_pose['y']:.2f}, {agent_pose['z']:.2f})"
            )

            # Process frame and update point cloud
            results = self.shepherd.process_frame(rgb_bgr, depth, agent_pose)
            self.last_frame_results = results
            self.last_frame_time = time.time()

        return {
            "rgb": rgb_bgr,
            "depth": depth,
            "results": self.last_frame_results,
            "camera_pose": agent_pose,
        }

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        self._reset_agent()
        return self._get_observation(), {}

    def render(self) -> np.ndarray:
        """Render environment with detections and depth visualization."""
        obs = self._get_observation()

        # Create visualization frame
        viz_frame = obs["rgb"].copy()

        # Draw detections and masks
        if obs["results"]:
            for result in obs["results"]:
                bbox = result["detection"]["bbox"]
                mask = result["mask"]
                object_id = result.get("object_id")
                similarity = result.get("similarity")

                # Get color based on whether we have a query
                if self.shepherd.config.default_query:
                    color = self.get_object_color(object_id, similarity)
                else:
                    color = self.get_object_color(object_id)

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
                cv2.putText(
                    viz_frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # Add query information
        query_text = f"Query: {self.shepherd.config.default_query}"
        cv2.putText(
            viz_frame,
            query_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Add depth visualization
        depth_viz = cv2.normalize(obs["depth"], None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

        # Combine RGB and depth
        combined_viz = np.hstack((viz_frame, depth_viz))

        return combined_viz

    def save_current_point_cloud(self):
        """Save current point cloud to PLY file."""
        output_path = os.path.join(self.output_dir, "habitat_point_cloud.ply")
        self.shepherd.database.save_point_cloud_ply(output_path)
        print(f"Saved point cloud to: {output_path}")

    def close(self):
        """Clean up resources."""
        if self.sim:
            self.sim.close()

    def save_debug_point_cloud(self, points: np.ndarray, color: tuple = (1, 0, 0)):
        """Save a point cloud for debugging with timestamp."""
        if len(points) == 0:
            print("No points to save!")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"debug_cloud_{timestamp}.ply")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Add colors
        colors = np.tile(color, (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save point cloud
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved debug point cloud to: {output_path}")


def main():
    # Initialize Shepherd with config
    config = ShepherdConfig(camera_height=1.0, camera_pitch=0.0)

    # Update camera parameters for Habitat
    config.camera = CameraUtils(
        width=256,
        height=256,
        fov=1.57,  # 90 degrees FOV
        camera_height=1.5,
        camera_pitch=0.0,
        camera_yaw=0.0,
        camera_roll=0.0,
        coordinate_frame="habitat",
    )

    # Print initial configuration
    print("\nInitial Configuration:")
    print(f"Camera FOV: {np.degrees(config.camera.fov):.1f} degrees")
    print(f"Camera height: {config.camera.camera_height:.2f}m")
    print(f"Camera angles (degrees):")
    print(f"  Pitch: {np.degrees(config.camera.camera_pitch):.1f}")
    print(f"  Yaw: {np.degrees(config.camera.camera_yaw):.1f}")
    print(f"  Roll: {np.degrees(config.camera.camera_roll):.1f}")

    shepherd = Shepherd(config=config)

    # Initialize environment
    scene_path = "./Replica-Dataset/data/apartment_2/mesh.ply"

    try:
        env = HabitatEnv(scene_path, shepherd)
        obs, _ = env.reset()

        print("\nControls:")
        print("W - Move forward")
        print("A - Turn left")
        print("D - Turn right")
        print("Q - Enter query")
        print("S - Save point cloud")
        print("ESC - Exit")

        while True:
            # Render and display
            frame = env.render()
            cv2.imshow("Habitat Demo", frame)

            # Handle input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("w"):
                action = 0
            elif key == ord("a"):
                action = 1
            elif key == ord("d"):
                action = 2
            elif key == ord("q"):
                query = input("\nEnter query: ")
                shepherd.update_query(query)
                continue
            elif key == ord("s"):
                env.save_current_point_cloud()
                continue
            elif key == 27:  # ESC
                break
            else:
                continue

            obs, _, _, _, _ = env.step(action)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
