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
                 similarity_threshold: float = 1.5,
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
        self.nn = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
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

            return point_cloud

        except Exception as e:
            print(f"Error in point cloud cleaning: {e}")
            return point_cloud

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
                                    nn_threshold: float = 0.05) -> float:
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
                    camera_pose: Optional[Dict] = None) -> Optional[str]:
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
            new_id = str(self.collection.count() if self.collection.count() is not None else 0)
            
            self.point_clouds[new_id] = cleaned_points
            self.collection.add(
                embeddings=[processed_embedding.tolist()],
                ids=[new_id],
                metadatas=[metadata]
            )
            return new_id

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
                voxel_size = 0.01  # 5cm voxels
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
        
        # Update in database
        self.collection.update(
            ids=[existing_id],
            embeddings=[avg_embedding.tolist()],
            metadatas=[new_metadata]
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
                similarity = self._compute_semantic_similarity(
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

    def _compute_semantic_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                          normalize_output: bool = True) -> float:
        """Compute cosine similarity between embeddings."""

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

        # Optionally normalize to [0,1]
        if normalize_output:
            similarity = similarity / 2 + 0.5
            
        return similarity

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

    def update_query(self, query_embedding: Optional[np.ndarray]):
        """Update the query embedding used for similarity computations."""
        self.query_embedding = query_embedding