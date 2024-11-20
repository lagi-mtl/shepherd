import numpy as np
import chromadb
from typing import Dict, List, Tuple, Optional
import torch
from sklearn.neighbors import NearestNeighbors

class DatabaseWrapper:
    def __init__(self, collection_name: str = "detection_embeddings"):
        """Initialize database wrapper with ChromaDB."""
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.initialize_collection()
        self.point_clouds = {}  # Store point clouds in memory since they're too large for ChromaDB

    def store_object(self, embedding: np.ndarray, metadata: Dict, point_cloud: Optional[np.ndarray] = None) -> str:
        """Store a new object or update existing one."""
        # Flatten metadata structure
        flat_metadata = {}
        
        for key, value in metadata.items():
            if isinstance(value, dict):
                # For nested dictionaries (like depth_info), flatten with prefix
                for sub_key, sub_value in value.items():
                    if sub_key != 'depth_map':  # Skip depth map
                        flat_metadata[f"{key}_{sub_key}"] = str(sub_value)
            elif key != 'mask':  # Skip mask as it's too large
                flat_metadata[key] = str(value)

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
                    depth_info[depth_key] = float(v)
                else:
                    try:
                        processed_metadata[k] = float(v)
                    except:
                        processed_metadata[k] = v
            
            if depth_info:
                processed_metadata['depth_info'] = depth_info
            
            # Get point cloud if available
            point_cloud = self.point_clouds.get(current_id, None)
            
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
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

    def initialize_collection(self):
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