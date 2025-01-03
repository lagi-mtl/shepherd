"""
BLIP model implementation.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from ..captioning_model import CaptioningModel


class BLIP(CaptioningModel):
    """
    BLIP model implementation.
    """

    def load_model(self):
        """Load BLIP model and processor."""
        model_name = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for BLIP."""
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        # Handle different image modes
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def generate_caption(self, image: np.ndarray) -> str:
        """Generate caption for image."""
        try:
            inputs = self.preprocess(image)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=50, num_beams=5)
            return self.postprocess(outputs)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            print(f"Error generating caption: {str(e)}")
            return ""

    def postprocess(self, output: torch.Tensor) -> str:
        """Convert output tokens to caption string."""
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()
