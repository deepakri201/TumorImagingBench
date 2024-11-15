from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import monai
from fmcib.preprocessing import SeedBasedPatchCropd


class BaseModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load(self, weights_path: str):
        """Load model weights from file"""
        pass

    @abstractmethod
    def preprocess(self, x):
        """Preprocess input data before forward pass"""
        pass

    @abstractmethod
    def forward(self, x):
        """Forward pass of the model"""
        pass


def get_transforms(
    orient="SPL",
    scale_range=(-1024, 2048),
    spatial_size=(48, 48, 48),
    spacing=(1, 1, 1),
):
    T = monai.transforms.Compose(
        [
            monai.transforms.CopyItemsd(keys=["image_path"], names=["image"]),
            monai.transforms.LoadImaged(
                keys=["image"], ensure_channel_first=True, reader="ITKReader"
            ),
            monai.transforms.EnsureTyped(keys=["image"]),
            monai.transforms.Spacingd(
                keys=["image"],
                pixdim=spacing,
                padding_mode="zeros",
                mode="linear",
                align_corners=True,
                diagonal=True,
            ),
            monai.transforms.Orientationd(keys=["image"], axcodes="LPS"),
            SeedBasedPatchCropd(
                keys=["image"],
                roi_size=spatial_size,
                coord_orientation="LPS",
                global_coordinates=True,
            ),
            monai.transforms.Orientationd(keys=["image"], axcodes=orient),
            monai.transforms.ScaleIntensityRanged(
                keys="image",
                a_min=scale_range[0],
                a_max=scale_range[1],
                b_min=0,
                b_max=1,
                clip=True,
            ),
            monai.transforms.SpatialPadd(keys=["image"], spatial_size=spatial_size),
            monai.transforms.Lambda(func=lambda x: x["image"].as_tensor()),
        ]
    )
    return T
