import os
import torch
import merlin
import monai.transforms
from fmcib.preprocessing import SeedBasedPatchCropd
from . import BaseModel, get_transforms


class MerlinExtractor(BaseModel):
    """Merlin model for extracting image and text embeddings"""

    def __init__(self):
        super().__init__()
        self.model = merlin.models.Merlin()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transforms = get_transforms(
            orient="RAS",
            scale_range=(-1000, 1000),
            spatial_size=(48, 48, 48),
            spacing=(1, 1, 1),
        )

    def load(self, weights_path: str = None):
        """Load pretrained weights"""
        pass

    def preprocess(self, x):
        """Apply transforms to input data"""
        return self.transforms(x)

    def forward(self, image, text=" "):
        """
        Forward pass to extract embeddings
        Args:
            image: Input image tensor
            text: Optional text input
        Returns:
            Image embeddings, phenotype predictions, and text embeddings (if text provided)
        """
        image = image.to(self.device)
        outputs = self.model(image, text)
        return outputs[0]
