import torch
import torch.nn as nn
import monai
from fmcib.preprocessing import SeedBasedPatchCropd
import wget
import os
from loguru import logger

from .utils import get_transforms
from .base import BaseModel

class VISTA3DExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = monai.networks.nets.segresnet_ds.SegResEncoder(
            spatial_dims=3,
            in_channels=1,
            init_filters=48,
            blocks_down=[1, 2, 2, 4, 4],
            norm="instance",
            head_module=lambda x: torch.nn.functional.adaptive_avg_pool3d(
                x[-1], 1
            ).flatten(
                start_dim=1
            ),  # Get only the last feature across block levels and average pool it.
        )
        self.transforms = get_transforms(
            orient="RAS",
            scale_range=(-1024, 2048),
            spatial_size=(48, 48, 48),
            spacing=(1, 1, 1),
        )

    def load(self, weights_path: str = None):
        # Download weights from huggingface if path not provided
        if weights_path is None:
            weights_path = "model_vista3d.pt"
            if not os.path.exists(weights_path):
                weights_path = wget.download(
                    "https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo/model_vista3d.pt",
                    bar=wget.bar_adaptive,
                )
        weights = torch.load(weights_path)
        # Modify prefix of weights to match model structure
        weights = {
            k.replace("image_encoder.encoder.", ""): v for k, v in weights.items()
        }
        msg = self.model.load_state_dict(
            weights, strict=False
        )  # Set strict to False as we load only the encoder

        logger.info(msg)
        self.model.eval()

    def preprocess(self, x):
        return self.transforms(x)

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
