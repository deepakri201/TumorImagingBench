import torch
import torch.nn as nn
import monai
from fmcib.preprocessing import SeedBasedPatchCropd
from . import BaseModel, get_transforms
from .unet3d import UNet3D
from huggingface_hub import hf_hub_download


class SUPREMExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = UNet3D(n_class=1)
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
                repo_id="MrGiovanni/SuPreM", filename="supervised_suprem_unet_2100.pth"
            )

        weights = torch.load(weights_path)
        model_dict = torch.load(weights_path)["net"]
        store_dict = self.model.state_dict()

        print("Loading SuPreM UNet backbone pretrained weights")
        decoder_keys = ["up_tr", "out_tr"]
        amount = 0
        for key in model_dict.keys():
            new_key = ".".join(key.split(".")[2:])
            if new_key in store_dict.keys():
                if any(decoder_key in new_key for decoder_key in decoder_keys):
                    continue
                store_dict[new_key] = model_dict[key]
                amount += 1

        print(f"Loaded {amount}/{len(store_dict.keys())} keys")
        # Modify prefix of weights to match model structure
        self.model.load_state_dict(store_dict)
        self.model = Encoder(self.model)
        self.model.eval()

    def preprocess(self, x):
        return self.transforms(x)

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super(Encoder, self).__init__()
        self.model = model

    def forward(self, x):
        self.out64, self.skip_out64 = self.model.down_tr64(x)
        self.out128, self.skip_out128 = self.model.down_tr128(self.out64)
        self.out256, self.skip_out256 = self.model.down_tr256(self.out128)
        self.out512 = self.model.down_tr512(self.out256)[0]
        return torch.nn.functional.adaptive_avg_pool3d(self.out512, 1).flatten(
            start_dim=1
        )  # Get only the last feature across block levels and average pool it.
