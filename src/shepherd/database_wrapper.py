import numpy as np
import chromadb
from typing import Dict, List, Tuple, Optional
import torch
from sklearn.neighbors import NearestNeighbors
from .utils.camera import CameraUtils

class DatabaseWrapper:
    def __init__(self, collection_name: str = "detection_embeddings", camera_utils: Optional[CameraUtils] = None):
        """Initialize database wrapper with ChromaDB."""
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.initialize_collection()
        self.point_clouds = {}  # Store point clouds in memory
        self.camera = camera_utils if camera_utils is not None else CameraUtils()
        
    def store_object(self, embedding: np.ndarray, metadata: Dict, 
                    point_cloud: Optional[np.ndarray] = None,
                    camera_pose: Optional[Dict] = None) -> str:
        """Store a new object or update existing one."""
        if camera_pose is None:
            camera_pose = {'x': 0, 'y': 0, 'z': 0, 'qx': 0, 'qy': 0, 'qz': 0, 'qw': 1}
            
        # Transform point cloud to world coordinates using camera utils
        if point_cloud is not None and len(point_cloud) > 0:
            point_cloud = self.camera.transform_point_cloud(point_cloud, camera_pose)
            
        # Check for nearby objects
        nearby_id = self._find_nearby_object(point_cloud, embedding)
        
        if nearby_id is not None:
            return self._merge_objects(nearby_id, embedding, metadata, point_cloud)
        else:
            # Flatten metadata structure
            flat_metadata = self._flatten_metadata(metadata)
            
            # Add new object
            new_id = str(self.collection.count() if self.collection.count() is not None else 0)
            
            # Store point cloud in memory if provided
            if point_cloud is not None and len(point_cloud) > 0:
                self.point_clouds[new_id] = point_cloud
            
            self.collection.add(
                embeddings=[embedding.tolist()],
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

    def _find_nearby_object(self, point_cloud: np.ndarray, embedding: np.ndarray,
                           distance_threshold: float = 0.5,
                           similarity_threshold: float = 0.8) -> Optional[str]:
        """Find nearby object based on point cloud position and embedding similarity."""
        if len(self.point_clouds) == 0 or point_cloud is None:
            return None
            
        # Get centroids of all stored point clouds
        centroids = {}
        for obj_id, cloud in self.point_clouds.items():
            centroids[obj_id] = np.mean(cloud, axis=0)
            
        # Get centroid of new point cloud
        new_centroid = np.mean(point_cloud, axis=0)
        
        # Check distances and similarities
        results = self.collection.get(include=['embeddings'])
        
        for obj_id, stored_embedding in zip(self.point_clouds.keys(), results['embeddings']):
            # Check distance between centroids
            distance = np.linalg.norm(new_centroid - centroids[obj_id])
            
            if distance < distance_threshold:
                # If objects are close, check embedding similarity
                similarity = self._compute_similarity(embedding, np.array(stored_embedding))
                if similarity > similarity_threshold:
                    return obj_id
                    
        return None

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
        
        # Merge point clouds with voxelization and occlusion handling
        if len(old_point_cloud) > 0 and len(new_point_cloud) > 0:
            merged_cloud = self._merge_point_clouds(old_point_cloud, new_point_cloud)
        else:
            merged_cloud = new_point_cloud if len(new_point_cloud) > 0 else old_point_cloud
            
        # Average embeddings
        avg_embedding = (new_embedding + old_embedding) / 2
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

    def _merge_point_clouds(self, cloud1: np.ndarray, cloud2: np.ndarray,
                           voxel_size: float = 0.05,
                           occlusion_threshold: float = 0.1) -> np.ndarray:
        """Merge point clouds with voxelization and occlusion handling."""
        # Create a set of unique points (rounded to reduce duplicates)
        unique_points = set()
        final_points = []
        
        # Process all points
        for point in np.vstack([cloud1, cloud2]):
            point_key = tuple(np.round(point / voxel_size))
            if point_key not in unique_points:
                unique_points.add(point_key)
                final_points.append(point)
        
        if not final_points:
            return np.array([])
            
        points = np.array(final_points)
        
        # Remove outliers
        if len(points) > 1:
            centroid = np.mean(points, axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            valid_points = distances < (np.mean(distances) + 2 * np.std(distances))
            points = points[valid_points]
            
        return points

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

    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        # Check for zero norms to avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

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