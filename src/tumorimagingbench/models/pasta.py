import torch
import torch.nn as nn
import monai
import fmcib
from huggingface_hub import hf_hub_download
from .PASTA.pasta_unet import Generic_UNet

from .utils import get_transforms
from .base import BaseModel

class PASTAExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = Generic_UNet(input_channels=1)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.transforms = get_transforms(
            orient="LPS", # Refer: https://github.com/LWHYC/PASTA/blob/main/preprocess/NifitiStandard.py
            scale_range=(-1024, 2048),
            spatial_size=(48, 48, 48),
            spacing=(1, 1, 1), # Refer: https://github.com/LWHYC/PASTA/blob/main/preprocess/NifitiStandard.py
        )

    def load(self, weights_path: str = None):
        # Download weights from huggingface if path not provided
        if weights_path is None:
            weights_path = hf_hub_download(
                repo_id="surajpaib/PASTA-Model",
                filename="PASTA final.pth",
            )

        weights = torch.load(weights_path)
        weights = {key.replace("module.", ""): value for key, value in weights.items()}        

        msg = self.model.load_state_dict(
            weights, strict=False
        )
        print("Model load message:", msg)
        self.model.eval()

    def preprocess(self, x):
        return self.transforms(x)

    def forward(self, x, avg=True):
        with torch.no_grad():
            out = self.model(x)
            return torch.nn.functional.adaptive_avg_pool3d(out, 1).flatten(start_dim=1) if avg else out

if __name__ == "__main__":
    # Create an instance of the extractor and load model weights
    extractor = ModelsGenExtractor()
    extractor.load()

    # Create a dummy 3D input tensor with shape (batch_size, channels, depth, height, width)
    dummy_input = torch.rand(1, 1, 48, 48, 48)

    # Optionally, preprocess the input if needed
    processed_input = extractor.preprocess(dummy_input)

    # Run inference using the forward method
    output = extractor.forward(processed_input)

    # Print the output details
    print("Output tensor shape:", output.shape)