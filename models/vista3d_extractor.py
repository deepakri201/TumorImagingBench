import torch
import torch.nn as nn
import monai 
from fmcib.preprocessing import SeedBasedPatchCropd
from . import BaseModel
import wget
import os


class VISTA3DExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = monai.networks.nets.segresnet_ds.SegResEncoder(
            spatial_dims=3,
            in_channels=1,
            init_filters=48,
            blocks_down=[1, 2, 2, 4, 4],
            norm="instance",
            head_module=lambda x: torch.nn.functional.adaptive_avg_pool3d(x[-1], 1).flatten(start_dim=1) # Get only the last feature across block levels and average pool it.
        )
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
                monai.transforms.Orientationd(keys=["image"], axcodes="RAS"),
                monai.transforms.ScaleIntensityRanged(keys="image", a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
                monai.transforms.SpatialPadd(keys=["image"], spatial_size=(48, 48, 48)),
                monai.transforms.Lambda(func=lambda x: x["image"].as_tensor())
            ])

    def load(self, weights_path: str = None):
        # Download weights from huggingface if path not provided
        if weights_path is None:
            weights_path = "model_vista3d.pt"
            if not os.path.exists(weights_path):
                weights_path = wget.download("https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo/model_vista3d.pt", bar=wget.bar_adaptive)
        weights = torch.load(weights_path)
        # Modify prefix of weights to match model structure
        weights = {k.replace('image_encoder.encoder.', ''): v for k, v in weights.items()}
        self.model.load_state_dict(weights, strict=False) # Set strict to False as we load only the encoder
        self.model.eval()

    def preprocess(self, x):
        return self.transforms(x)

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
