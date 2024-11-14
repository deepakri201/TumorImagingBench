import torch
import torch.nn as nn
import monai 
from fmcib.preprocessing import SeedBasedPatchCropd
from . import BaseModel
from huggingface_hub import hf_hub_download


class VocoExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = monai.networks.nets.SwinUNETR(
            img_size=(64, 64, 64),
            in_channels=1,
            out_channels=2,
            feature_size=192,
            use_v2=True
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
            monai.transforms.SpatialPadd(keys=["image"], spatial_size=(64, 64, 64)),
            monai.transforms.Lambda(func=lambda x: x["image"].as_tensor())
        ])

    def load(self, weights_path: str = None):
        if weights_path is None:
            weights_path = hf_hub_download(
                repo_id="Luffy503/VoCo",
                filename="VoCo_H_SSL_head.pt"
            )
        
        weights = torch.load(weights_path)
        if "state_dict" in weights:
            weights = weights["state_dict"]
        
        current_model_dict = self.model.state_dict()
        new_state_dict = {
            k: weights[k] if k in weights and weights[k].size() == current_model_dict[k].size() else current_model_dict[k]
            for k in current_model_dict
        }
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.eval()
        self.model = self.model.swinViT

    def preprocess(self, x):
        return self.transforms(x)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
            return torch.nn.functional.adaptive_avg_pool3d(features[-1], 1).flatten(start_dim=1)
