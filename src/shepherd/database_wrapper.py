import numpy as np
import chromadb
from typing import Dict, List, Tuple, Optional
import torch
from sklearn.neighbors import NearestNeighbors
from .utils.camera import CameraUtils
from sklearn.cluster import DBSCAN

class DatabaseWrapper:
    def __init__(self, collection_name: str = "detection_embeddings", 
                 camera_utils: Optional[CameraUtils] = None,
                 distance_threshold: float = 1.5,
                 similarity_threshold: float = 0.5,
                 cluster_eps: float = 0.2,
                 cluster_min_samples: int = 3):
        """Initialize database wrapper with ChromaDB."""
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.initialize_collection()
        self.point_clouds = {}  # Store point clouds in memory
        self.camera = camera_utils if camera_utils is not None else CameraUtils()
        
        # Thresholds for object matching and merging
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        
        # Initialize nearest neighbors
        self.nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        self.query_embedding = None  # Add query embedding storage

    def _clean_point_cloud(self, point_cloud: np.ndarray, mask: np.ndarray = None, camera_pose: Optional[Dict] = None) -> np.ndarray:
        """Clean point cloud using nearest neighbors and line-of-sight filtering."""
        if point_cloud is None or len(point_cloud) < self.cluster_min_samples:
            return np.array([])

        try:
            # Remove NaN and infinite values
            valid_mask = np.all(np.isfinite(point_cloud), axis=1)
            point_cloud = point_cloud[valid_mask]
            
            if len(point_cloud) < self.cluster_min_samples:
                return point_cloud  # Return original if too few points

            # Use nearest neighbors to find core points
            self.nn.fit(point_cloud)
            distances, _ = self.nn.kneighbors(point_cloud)
            core_points_mask = distances.ravel() < self.cluster_eps
            core_points = point_cloud[core_points_mask]

            if len(core_points) == 0:
                return point_cloud  # Return original if no core points found

            # Line of sight filtering
            if camera_pose is not None:
                camera_pos = np.array([camera_pose['x'], camera_pose['y'], camera_pose['z']])
                
                # Sort points by distance from camera
                directions = point_cloud - camera_pos
                distances = np.linalg.norm(directions, axis=1)
                sorted_indices = np.argsort(distances)
                
                # Keep points that are not occluded
                valid_points = []
                for idx in sorted_indices:
                    point = point_cloud[idx]
                    direction = directions[idx]
                    distance = distances[idx]
                    
                    # Check if this point is occluded by any previously accepted point
                    occluded = False
                    for valid_point in valid_points:
                        if self._is_occluded(camera_pos, point, valid_point, threshold=0.1):
                            occluded = True
                            break
                    
                    if not occluded:
                        valid_points.append(point)
                
                point_cloud = np.array(valid_points)

            return point_cloud

        except Exception as e:
            print(f"Error in point cloud cleaning: {e}")
            return point_cloud

    def _is_occluded(self, camera_pos: np.ndarray, point: np.ndarray, 
                     obstacle: np.ndarray, threshold: float = 0.1) -> bool:
        """Check if a point is occluded by an obstacle from camera perspective."""
        # Vector from camera to point
        dir_to_point = point - camera_pos
        dist_to_point = np.linalg.norm(dir_to_point)
        dir_to_point = dir_to_point / dist_to_point

        # Vector from camera to obstacle
        dir_to_obstacle = obstacle - camera_pos
        dist_to_obstacle = np.linalg.norm(dir_to_obstacle)
        dir_to_obstacle = dir_to_obstacle / dist_to_obstacle

        # If obstacle is further than point, it can't occlude
        if dist_to_obstacle > dist_to_point:
            return False

        # Check if directions are similar (potential occlusion)
        angle = np.arccos(np.clip(np.dot(dir_to_point, dir_to_obstacle), -1.0, 1.0))
        return angle < threshold

    def _process_new_detection(self, point_cloud: np.ndarray, embedding: np.ndarray, 
                              camera_pose: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Process new detection with DBSCAN clustering and pose transformation."""
        if point_cloud is None or len(point_cloud) < self.cluster_min_samples:
            return np.array([]), embedding

        try:
            # Transform to world coordinates
            point_cloud = self.camera.transform_point_cloud(point_cloud, camera_pose)
            
            # DBSCAN clustering for noise removal
            clustering = DBSCAN(
                eps=self.cluster_eps,
                min_samples=self.cluster_min_samples,
                n_jobs=-1
            ).fit(point_cloud)
            
            # Get largest cluster
            labels = clustering.labels_
            unique_labels = np.unique(labels)
            if len(unique_labels) == 1 and unique_labels[0] == -1:
                return np.array([]), embedding
            
            # Get points from largest non-noise cluster
            cluster_sizes = np.array([np.sum(labels == label) for label in unique_labels if label != -1])
            if len(cluster_sizes) == 0:
                return np.array([]), embedding
            
            main_cluster_label = unique_labels[np.argmax(cluster_sizes)]
            cleaned_points = point_cloud[labels == main_cluster_label]
            
            return cleaned_points, embedding
            
        except Exception as e:
            print(f"Error in detection processing: {e}")
            return np.array([]), embedding

    def _compute_geometric_similarity(self, points1: np.ndarray, points2: np.ndarray, 
                                    nn_threshold: float = 0.1) -> float:
        """Compute geometric similarity using nearest neighbor ratio."""
        if len(points1) == 0 or len(points2) == 0:
            return 0.0
        
        try:
            # Build KD-tree for second point cloud
            tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points2)
            
            # Find nearest neighbors for all points in first cloud
            distances, _ = tree.kneighbors(points1)
            
            # Count points with neighbors within threshold
            close_points = np.sum(distances < nn_threshold)
            
            # Return ratio of close points to total points
            return close_points / len(points1)
            
        except Exception as e:
            print(f"Error in geometric similarity computation: {e}")
            return 0.0

    def _compute_semantic_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute normalized cosine similarity between embeddings."""
        try:
            similarity = self._compute_similarity(embedding1, embedding2)
            # Normalize to [0,1] range as per paper
            return similarity / 2 + 0.5
        except Exception as e:
            print(f"Error in semantic similarity computation: {e}")
            return 0.0

    def _find_nearby_object(self, point_cloud: np.ndarray, embedding: np.ndarray) -> Optional[str]:
        """Find nearby object using combined geometric and semantic similarity."""
        if len(self.point_clouds) == 0 or point_cloud is None or len(point_cloud) == 0:
            return None
        
        try:
            best_match = None
            best_similarity = -float('inf')
            
            # Get embeddings from database
            results = self.collection.get(include=['embeddings'])
            
            # Check each existing object
            for obj_id, stored_cloud in self.point_clouds.items():
                if len(stored_cloud) == 0:
                    continue
                
                # Get stored embedding
                stored_embedding = results['embeddings'][list(self.point_clouds.keys()).index(obj_id)]
                
                # Compute geometric similarity
                geo_sim = self._compute_geometric_similarity(point_cloud, stored_cloud)
                
                # Only compute semantic similarity if there's geometric overlap
                if geo_sim > 0:
                    # Compute semantic similarity
                    sem_sim = self._compute_semantic_similarity(embedding, np.array(stored_embedding))
                    
                    # Combined similarity score
                    total_sim = geo_sim + sem_sim
                    
                    if total_sim > best_similarity:
                        best_similarity = total_sim
                        best_match = obj_id
            
            # Return match only if similarity is above threshold
            if best_similarity > self.similarity_threshold:
                return best_match
            
            return None
            
        except Exception as e:
            print(f"Error in finding nearby object: {e}")
            return None

    def _calculate_bbox_overlap(self, bbox1: Tuple[np.ndarray, np.ndarray], 
                              bbox2: Tuple[np.ndarray, np.ndarray]) -> float:
        """Calculate overlap between two 3D bounding boxes."""
        min1, max1 = bbox1
        min2, max2 = bbox2
        
        # Calculate intersection
        intersection_min = np.maximum(min1, min2)
        intersection_max = np.minimum(max1, max2)
        
        if np.any(intersection_max < intersection_min):
            return 0.0
        
        # Calculate volumes
        intersection_volume = np.prod(intersection_max - intersection_min)
        volume1 = np.prod(max1 - min1)
        volume2 = np.prod(max2 - min2)
        
        # Calculate IoU (Intersection over Union)
        union_volume = volume1 + volume2 - intersection_volume
        if union_volume <= 0:
            return 0.0
        
        return intersection_volume / union_volume

    def _merge_point_clouds(self, cloud1: np.ndarray, cloud2: np.ndarray,
                           voxel_size: float = 0.05) -> np.ndarray:
        """Merge point clouds with improved cleaning and voxelization."""
        if len(cloud1) == 0:
            return cloud2
        if len(cloud2) == 0:
            return cloud1
        
        # Combine clouds
        combined_cloud = np.vstack([cloud1, cloud2])
        
        # Remove NaN and infinite values
        valid_mask = np.all(np.isfinite(combined_cloud), axis=1)
        combined_cloud = combined_cloud[valid_mask]
        
        if len(combined_cloud) == 0:
            return np.array([])
        
        # Clean combined cloud
        cleaned_cloud = self._clean_point_cloud(combined_cloud)
        
        if len(cleaned_cloud) == 0:
            return np.array([])
        
        # Voxelize the cleaned cloud
        try:
            voxel_dict = {}
            for point in cleaned_cloud:
                voxel_key = tuple(np.round(point / voxel_size))
                if voxel_key not in voxel_dict:
                    voxel_dict[voxel_key] = []
                voxel_dict[voxel_key].append(point)
            
            # Compute centroids for each voxel
            voxelized_points = []
            for points in voxel_dict.values():
                if len(points) > 0:
                    centroid = np.median(points, axis=0)  # Use median for robustness
                    voxelized_points.append(centroid)
            
            return np.array(voxelized_points)
        except Exception as e:
            print(f"Error in point cloud merging: {e}")
            return cleaned_cloud  # Return cleaned cloud if voxelization fails

    def store_object(self, embedding: np.ndarray, metadata: Dict, 
                    point_cloud: Optional[np.ndarray] = None,
                    camera_pose: Optional[Dict] = None,
                    mask: Optional[np.ndarray] = None) -> Optional[str]:
        """Store a new object or update existing one."""
        if camera_pose is None:
            camera_pose = {'x': 0, 'y': 0, 'z': 0, 'qx': 0, 'qy': 0, 'qz': 0, 'qw': 1}
            
        # Process new detection
        cleaned_points, processed_embedding = self._process_new_detection(
            point_cloud, embedding, camera_pose
        )
        
        if len(cleaned_points) == 0:
            return None
        
        # Find nearby object
        nearby_id = self._find_nearby_object(cleaned_points, processed_embedding)
        
        if nearby_id is not None:
            return self._merge_objects(nearby_id, processed_embedding, metadata, cleaned_points)
        else:
            # Add new object
            flat_metadata = self._flatten_metadata(metadata)
            new_id = str(self.collection.count() if self.collection.count() is not None else 0)
            
            self.point_clouds[new_id] = cleaned_points
            self.collection.add(
                embeddings=[processed_embedding.tolist()],
                ids=[new_id],
                metadatas=[flat_metadata]
            )
            return new_id

    def _flatten_metadata(self, metadata: Dict) -> Dict:
        """Flatten nested metadata structure for ChromaDB storage."""
        flat_metadata = {}
        
        for key, value in metadata.items():
            if isinstance(value, dict):
                # For nested dictionaries (like depth_info), flatten with prefix
                for sub_key, sub_value in value.items():
                    if sub_key != 'depth_map':  # Skip depth map
                        flat_metadata[f"{key}_{sub_key}"] = str(sub_value)
            elif key != 'mask':  # Skip mask as it's too large
                flat_metadata[key] = str(value)
                
        return flat_metadata

    def _merge_objects(self, existing_id: str, new_embedding: np.ndarray,
                      new_metadata: Dict, new_point_cloud: np.ndarray) -> str:
        """Merge new object with existing one."""
        # Get existing object data
        results = self.collection.get(
            ids=[existing_id],
            include=['embeddings', 'metadatas']
        )
        
        old_embedding = np.array(results['embeddings'][0])
        old_point_cloud = self.point_clouds.get(existing_id, np.array([]))
        
        # Merge point clouds with voxelization
        if len(old_point_cloud) > 0 and len(new_point_cloud) > 0:
            # Combine clouds
            combined_cloud = np.vstack([old_point_cloud, new_point_cloud])
            
            # Remove duplicates and outliers
            if len(combined_cloud) > 0:
                # Voxelize to remove duplicates
                voxel_size = 0.05  # 5cm voxels
                voxel_dict = {}
                for point in combined_cloud:
                    voxel_key = tuple(np.round(point / voxel_size))
                    if voxel_key not in voxel_dict:
                        voxel_dict[voxel_key] = []
                    voxel_dict[voxel_key].append(point)
                
                # Get centroids of voxels
                merged_cloud = np.array([
                    np.mean(points, axis=0) 
                    for points in voxel_dict.values()
                ])
                
                # Remove statistical outliers
                if len(merged_cloud) > 0:
                    centroid = np.median(merged_cloud, axis=0)
                    distances = np.linalg.norm(merged_cloud - centroid, axis=1)
                    inliers = distances < (np.median(distances) + 2 * np.std(distances))
                    merged_cloud = merged_cloud[inliers]
        else:
            merged_cloud = new_point_cloud if len(new_point_cloud) > 0 else old_point_cloud
        
        # Update embedding with weighted average (based on point cloud sizes)
        weight_old = len(old_point_cloud) / (len(old_point_cloud) + len(new_point_cloud))
        weight_new = 1 - weight_old
        avg_embedding = (old_embedding * weight_old + new_embedding * weight_new)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        # Update metadata
        flat_metadata = self._flatten_metadata(new_metadata)
        
        # Update in database
        self.collection.update(
            ids=[existing_id],
            embeddings=[avg_embedding.tolist()],
            metadatas=[flat_metadata]
        )
        
        # Update point cloud
        self.point_clouds[existing_id] = merged_cloud
        
        return existing_id

    def query_objects(self, query_text: str, clip_model = None) -> List[Dict]:
        """Query objects in database using text query."""
        if self.collection.count() == 0:
            return []
            
        # Get all objects
        results = self.collection.get(include=['metadatas', 'embeddings'])
        query_results = []
        
        # Encode query text if CLIP model provided
        if clip_model is not None:
            query_embedding = clip_model.encode_text(query_text)
        
        # Get all IDs from collection
        all_ids = [str(i) for i in range(self.collection.count())]
        
        for i, (metadata, embedding) in enumerate(zip(results['metadatas'], results['embeddings'])):
            if metadata is None or any(k.startswith('dummy') for k in metadata.keys()):
                continue
                
            # Get current ID
            current_id = all_ids[i]
                
            # Compute similarity if CLIP model provided
            similarity = 0.0
            if clip_model is not None:
                similarity = self._compute_similarity(
                    np.array(embedding),
                    query_embedding
                )
            
            # Reconstruct nested metadata structure
            processed_metadata = {}
            depth_info = {}
            
            for k, v in metadata.items():
                if k.startswith('depth_info_'):
                    # Extract depth info fields
                    depth_key = k.replace('depth_info_', '')
                    try:
                        depth_info[depth_key] = float(v)
                    except ValueError:
                        depth_info[depth_key] = v
                else:
                    try:
                        processed_metadata[k] = float(v)
                    except ValueError:
                        processed_metadata[k] = v
            
            if depth_info:
                processed_metadata['depth_info'] = depth_info
            
            # Get point cloud if available
            point_cloud = self.point_clouds.get(current_id, None)
            
            # Only add results with valid point clouds
            if point_cloud is not None and len(point_cloud) > 0:
                query_results.append({
                    'similarity': similarity,
                    'metadata': processed_metadata,
                    'embedding': np.array(embedding),
                    'point_cloud': point_cloud
                })
            
        # Sort results by similarity
        query_results.sort(key=lambda x: x['similarity'], reverse=True)
        return query_results

    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                          normalize_output: bool = False) -> float:
        """Compute cosine similarity between embeddings."""
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0
            
            # Convert to numpy arrays if needed
            if isinstance(embedding1, torch.Tensor):
                embedding1 = embedding1.detach().cpu().numpy()
            if isinstance(embedding2, torch.Tensor):
                embedding2 = embedding2.detach().cpu().numpy()
            
            embedding1 = np.array(embedding1)
            embedding2 = np.array(embedding2)
            
            # Ensure embeddings are normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            embedding1 = embedding1 / norm1
            embedding2 = embedding2 / norm2
            
            similarity = float(np.dot(embedding1, embedding2))
            
            # Optionally normalize to [0,1] range as per paper
            if normalize_output:
                similarity = similarity / 2 + 0.5
                
            return similarity
        except Exception as e:
            print(f"Error in similarity computation: {e}")
            return 0.0

    def initialize_collection(self): # TODO: Fix this to not use dummy vectors
        """Initialize collection with dummy vectors."""
        if self.collection.count() == 0:
            dummy_vectors = np.zeros((3, 512))
            self.collection.add(
                embeddings=dummy_vectors.tolist(),
                ids=["dummy_1", "dummy_2", "dummy_3"],
                metadatas=[
                    {"dummy": "1"},
                    {"dummy": "2"},
                    {"dummy": "3"}
                ]
            )

    def _create_point_cloud(self, mask: np.ndarray, depth_frame: np.ndarray) -> np.ndarray:
        """Create point cloud from mask and depth frame."""
        # Get image dimensions
        height, width = depth_frame.shape
        
        # Create meshgrid of pixel coordinates
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        
        # Apply mask
        valid_points = mask > 0
        
        # Get valid coordinates and depths
        x = xx[valid_points]
        y = yy[valid_points]
        z = depth_frame[valid_points]
        
        # Filter out invalid depths
        valid_depths = z > 0.1
        x = x[valid_depths]
        y = y[valid_depths]
        z = z[valid_depths]
        
        # Convert to 3D coordinates using camera parameters
        # Note: These should match the camera parameters in ShepherdConfig
        fx = 1344 / (2 * np.tan(1.88 / 2))  # width/(2*tan(fov/2))
        fy = fx
        cx = 1344 / 2
        cy = 376 / 2
        
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        Z = z
        
        # Stack coordinates
        points = np.stack([X, Y, Z], axis=1)
        
        # Remove outliers
        if len(points) > 0:
            mean = np.mean(points, axis=0)
            std = np.std(points, axis=0)
            valid_points = np.all(np.abs(points - mean) <= 2 * std, axis=1)
            points = points[valid_points]
        
        return points

    def update_query(self, query_embedding: Optional[np.ndarray]):
        """Update the query embedding used for similarity computations."""
        self.query_embedding = query_embedding