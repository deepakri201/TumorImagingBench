import os
import torch
import merlin
import monai.transforms
from fmcib.preprocessing import SeedBasedPatchCropd
from . import BaseModel

class MerlinExtractor(BaseModel):
    """Merlin model for extracting image and text embeddings"""
    
    def __init__(self):
        super().__init__()
        self.model = merlin.models.Merlin()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transforms = monai.transforms.Compose([
            monai.transforms.CopyItemsd(keys=["image_path"], names=["image"]),
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
            monai.transforms.Orientationd(keys=["image"], axcodes="RAS"),
            monai.transforms.Spacingd(
                keys=["image"], pixdim=1, padding_mode="zeros", mode="linear", align_corners=True, diagonal=True
            ),
            monai.transforms.Orientationd(keys=["image"], axcodes="LPS"),
            SeedBasedPatchCropd(
                keys=["image"], roi_size=(48, 48, 48), coord_orientation="LPS", global_coordinates=True
            ),    
            monai.transforms.Orientationd(keys=["image"], axcodes="RAS"),
            monai.transforms.ScaleIntensityRanged(keys="image", a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True),
            monai.transforms.SpatialPadd(keys=["image"], spatial_size=(48, 48, 48)),
            monai.transforms.Lambda(func=lambda x: x["image"].as_tensor())
        ])

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
