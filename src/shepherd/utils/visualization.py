"""
Visualization utilities for the Shepherd project.
"""

from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


class VisualizationUtils:
    """
    Visualization utilities for the Shepherd project to visualize intermediate results.
    """

    # YOLO class names mapping
    YOLO_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
    }

    @staticmethod
    def show_image(image: np.ndarray, title: str = "Image", figsize: tuple = (10, 10)):
        """Display a single image."""
        plt.figure(figsize=figsize)
        if len(image.shape) == 3 and image.shape[2] == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()

    @staticmethod
    def show_detections(
        image: np.ndarray,
        detections: List[Dict],
        show_labels: bool = True,
        show_conf: bool = True,
    ):
        """Display image with YOLO detections."""
        viz = image.copy()

        for det in detections:
            bbox = det["bbox"]
            conf = det.get("confidence", 0)
            class_id = int(det.get("class_id", 0))
            class_name = VisualizationUtils.YOLO_CLASSES.get(
                class_id, f"Class {class_id}"
            )

            x_1, y_1, x_2, y_2 = map(int, bbox)
            cv2.rectangle(viz, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)

            if show_labels or show_conf:
                label = []
                if show_labels:
                    label.append(class_name)
                if show_conf:
                    label.append(f"{conf:.2f}")

                label_text = " | ".join(label)
                cv2.putText(
                    viz,
                    label_text,
                    (x_1, y_1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
        plt.title("YOLO Detections")
        plt.axis("off")
        plt.show()

    @staticmethod
    def show_masks(image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.5):
        """Display image with segmentation masks."""
        viz = image.copy()

        # Create random colors for each mask
        colors = np.random.randint(0, 255, (len(masks), 3))

        # Create mask overlay
        mask_overlay = np.zeros_like(image)
        for i, mask in enumerate(masks):
            mask_overlay[mask] = colors[i]

        # Blend with original image
        cv2.addWeighted(mask_overlay, alpha, viz, 1 - alpha, 0, viz)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
        plt.title("Segmentation Masks")
        plt.axis("off")
        plt.show()

    @staticmethod
    def show_depth(depth_map: np.ndarray):
        """Display depth map."""
        if depth_map is None or not isinstance(depth_map, np.ndarray):
            print("No valid depth map to display")
            return

        plt.figure(figsize=(10, 10))
        plt.imshow(depth_map, cmap="plasma")
        plt.colorbar(label="Depth")
        plt.title("Depth Map")
        plt.axis("off")
        plt.show()

    @staticmethod
    def show_point_cloud(points: np.ndarray, colors: Optional[np.ndarray] = None):
        """Display 3D point cloud using Open3D."""
        if len(points) == 0:
            print("No points to display")
            return

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            if colors.shape[1] == 3:  # RGB colors
                pcd.colors = o3d.utility.Vector3dVector(colors)
            else:  # Single channel colors
                colors_rgb = np.tile(colors[:, np.newaxis], (1, 3))
                pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
        else:
            # Default color (blue)
            pcd.paint_uniform_color([0, 0, 1])

        # Calculate coordinate frame size based on point cloud dimensions
        points_max = np.max(points, axis=0)
        points_min = np.min(points, axis=0)
        scale = np.max(points_max - points_min) * 0.2

        # Create coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=scale, origin=[0, 0, 0]
        )

        # Estimate normals for better visualization
        pcd.estimate_normals()

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add geometry
        vis.add_geometry(pcd)
        vis.add_geometry(coord_frame)

        # Set view control
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])  # Black background
        opt.point_size = 2.0

        # Set initial viewpoint
        vc = vis.get_view_control()
        vc.set_front([0, 0, -1])  # Look from front
        vc.set_up([0, -1, 0])  # Up direction
        vc.set_zoom(0.8)

        # Run visualizer
        vis.run()
        vis.destroy_window()

    @staticmethod
    def show_pipeline_step(
        step_name: str,
        image: np.ndarray,
        detections: Optional[List[Dict]] = None,
        masks: Optional[List[np.ndarray]] = None,
        depth: Optional[np.ndarray] = None,
        point_cloud: Optional[np.ndarray] = None,
    ):
        """Display results from a specific pipeline step."""
        print(f"\n=== {step_name} ===")

        if detections is not None:
            print(f"Found {len(detections)} detections")
            VisualizationUtils.show_detections(image, detections)

        if masks is not None:
            print(f"Generated {len(masks)} masks")
            VisualizationUtils.show_masks(image, masks)

        if depth is not None:
            print("Depth estimation:")
            VisualizationUtils.show_depth(depth)

        if point_cloud is not None:
            print(f"Point cloud with {len(point_cloud)} points:")
            VisualizationUtils.show_point_cloud(point_cloud)

    @staticmethod
    def show_query_results(query_results: List[Dict], query_text: str):
        """Display query results with color-coded point clouds in their actual 3D positions."""
        if not query_results:
            print("No results found")
            return

        try:
            # Create Open3D visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"Query Results: '{query_text}'")

            # Create coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0]
            )
            vis.add_geometry(coord_frame)

            # Print similarities and create point clouds
            print(f"\nQuery results for: '{query_text}'")

            # Get similarity range for normalization
            similarities = [r["similarity"] for r in query_results]
            min_similarity = min(similarities)
            similarity_range = max(similarities) - min_similarity

            # Combined point cloud for visualization scale
            all_points = []

            # Add each point cloud with color based on normalized similarity
            for i, result in enumerate(query_results):
                similarity = result["similarity"]
                metadata = result["metadata"]
                point_cloud = result.get("point_cloud", None)

                if point_cloud is None or len(point_cloud) == 0:
                    continue

                # Create point cloud
                pcd = o3d.geometry.PointCloud()

                # Use points in their original positions from depth estimation
                pcd.points = o3d.utility.Vector3dVector(point_cloud)
                all_points.extend(point_cloud)

                # Normalize similarity to [0, 1] range
                if similarity_range > 0:
                    normalized_similarity = (
                        similarity - min_similarity
                    ) / similarity_range
                else:
                    normalized_similarity = 0

                # Color based on similarity (white to red)
                # Higher similarity = more red
                color = np.array(
                    [1.0, 1.0 - normalized_similarity, 1.0 - normalized_similarity]
                )
                pcd.paint_uniform_color(color)

                # Add to visualizer
                vis.add_geometry(pcd)

                class_id = int(float(metadata.get("class_id", 0)))

                # Print metadata and similarity
                print(
                    f"\nObject {i+1} with similarity {similarity:.3f} "
                    f"(normalized: {normalized_similarity:.3f}):"
                )
                print(f"  Caption: {metadata.get('caption', 'No caption')}")
                print(
                    f"  Class: {VisualizationUtils.YOLO_CLASSES.get(class_id, 'Unknown')}"
                )
                print(f"  Confidence: {float(metadata.get('confidence', 0)):.2f}")

            # Calculate visualization bounds from all points
            if all_points:
                all_points = np.array(all_points)
                center = np.mean(all_points, axis=0)
                scale = np.max(np.abs(all_points - center)) * 2

                # Update coordinate frame size and position
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=scale * 0.2, origin=center
                )
                vis.add_geometry(coord_frame)

            # Set view control
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])  # Black background
            opt.point_size = 3.0

            # Run visualizer
            vis.run()
            vis.destroy_window()

        except Exception as e:
            print(f"Error in visualization: {e}")
            # Fallback to text-only display
            print("\nText-only results:")
            for result in query_results:
                print(f"\nSimilarity: {result['similarity']:.3f}")
                for k, v in result["metadata"].items():
                    if k not in ["mask", "point_cloud"]:
                        print(f"  {k}: {v}")
