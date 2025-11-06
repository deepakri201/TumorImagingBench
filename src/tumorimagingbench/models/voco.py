import torch
import monai
from huggingface_hub import hf_hub_download
from loguru import logger

from .utils import get_transforms
from .base import BaseModel

class VocoExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = monai.networks.nets.SwinUNETR(
            in_channels=1,
            out_channels=2,
            feature_size=192,
            use_v2=True,
        )
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.transforms = get_transforms(
            orient="RAS",
            scale_range=(-1024, 2048),
            spatial_size=(64, 64, 64),
            spacing=(1, 1, 1),
        )

    def load(self, weights_path: str = None):
        if weights_path is None:
            weights_path = hf_hub_download(
                repo_id="Luffy503/VoCo", filename="VoCo_H_SSL_head.pt"
            )

        weights = torch.load(weights_path)
        if "state_dict" in weights:
            weights = weights["state_dict"]

        current_model_dict = self.model.state_dict()
        new_state_dict = {
            k: (
                weights[k]
                if k in weights and weights[k].size() == current_model_dict[k].size()
                else current_model_dict[k]
            )
            for k in current_model_dict
        }
        msg = self.model.load_state_dict(new_state_dict, strict=True)
        logger.info(msg)
        self.model.eval()
        self.model = self.model.swinViT

    def preprocess(self, x):
        return self.transforms(x)

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
            return torch.nn.functional.adaptive_avg_pool3d(features[-1], 1).flatten(
                start_dim=1
            )
