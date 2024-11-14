import torch
import torch.nn as nn
import monai 
from fmcib.preprocessing import SeedBasedPatchCropd
from . import BaseModel
from fmcib.models import fmcib_model


class FMCIBExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = fmcib_model(eval_mode=False) # By default the model is in eval mode. Set to false if you want to train it 
        self.transforms = monai.transforms.Compose([
                monai.transforms.CopyItemsd(keys=["image_path"], names=["image"]),
                monai.transforms.LoadImaged(keys=["image"], ensure_channel_first=True, reader="ITKReader"),
                monai.transforms.EnsureTyped(keys=["image"]),
                monai.transforms.Spacingd(
                    keys=["image"], pixdim=1, padding_mode="zeros", mode="linear", align_corners=True, diagonal=True
                ),
                monai.transforms.Orientationd(keys=["image"], axcodes="LPS"),
                SeedBasedPatchCropd(
                    keys=["image"], roi_size=(48, 48, 48), coord_orientation="LPS", global_coordinates=True
                ),    
                monai.transforms.ScaleIntensityRanged(keys="image", a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
                monai.transforms.SpatialPadd(keys=["image"], spatial_size=(48, 48, 48)),
                monai.transforms.Lambda(func=lambda x: x["image"].as_tensor())
            ])

    def load(self, weights_path: str = None):
        pass

    def preprocess(self, x):
        return self.transforms(x)

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
