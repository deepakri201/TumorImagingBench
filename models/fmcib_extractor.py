import torch
import torch.nn as nn
import monai
from fmcib.preprocessing import SeedBasedPatchCropd
from . import BaseModel, get_transforms
from fmcib.models import fmcib_model


class FMCIBExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = fmcib_model(
            eval_mode=False
        )  # By default the model is in eval mode. Set to false if you want to train it
        self.transforms = get_transforms(
            orient="LPS",
            scale_range=(-1024, 2048),
            spatial_size=(48, 48, 48),
            spacing=(1, 1, 1),
        )

    def load(self, weights_path: str = None):
        pass

    def preprocess(self, x):
        return self.transforms(x)

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
