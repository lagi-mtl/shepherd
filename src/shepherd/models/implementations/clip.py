import torch
import clip
from PIL import Image
import numpy as np
from typing import List
from ..embedding_model import EmbeddingModel
import cv2

class CLIP(EmbeddingModel):
    def load_model(self):
        """Load CLIP model."""
        self.model, self.preprocess_fn = clip.load("ViT-B/32", device=self.device)
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for CLIP."""
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        return self.preprocess_fn(image).unsqueeze(0).to(self.device)
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Encode image to embedding vector."""
        try:
            image_input = self.preprocess(image)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return self.postprocess(image_features)
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            return np.zeros((512,))  # CLIP ViT-B/32 has 512-dimensional embeddings
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return self.postprocess(text_features)
    
    def compute_similarity(self, image_features: np.ndarray, text_features: np.ndarray) -> float:
        """Compute cosine similarity between image and text features."""
        image_features = torch.from_numpy(image_features).to(self.device)
        text_features = torch.from_numpy(text_features).to(self.device)
        similarity = torch.cosine_similarity(image_features, text_features).item()
        return similarity
    
    def postprocess(self, output: torch.Tensor) -> np.ndarray:
        """Convert output to numpy array."""
        return output.cpu().numpy().squeeze() 