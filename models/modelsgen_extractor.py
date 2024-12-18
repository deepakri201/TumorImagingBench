import torch
import torch.nn as nn
import monai
import fmcib
from . import BaseModel, get_transforms
from huggingface_hub import hf_hub_download

class ModelsGenExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = fmcib.models.ModelsGenesisUNet3D(decoder=False)
        self.transforms = get_transforms(
            orient="RAS",
            scale_range=(-1024, 2048),
            spatial_size=(48, 48, 48),
            spacing=(1, 1, 1),
        )

    
    def load(self, weights_path: str = None):
        # Download weights from huggingface if path not provided
        if weights_path is None:
            weights_path = hf_hub_download(
                repo_id="MrGiovanni/ModelsGenesis",
                filename="Genesis_Chest_CT.pt",
            )

        weights = torch.load(weights_path)
        weights = weights["state_dict"]
        weights = {key.replace("module.", ""): value for key, value in weights.items()}        

        msg = self.model.load_state_dict(
            weights, strict=False
        )
        print(msg)
        self.model.eval()

    def preprocess(self, x):
        return self.transforms(x)

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

