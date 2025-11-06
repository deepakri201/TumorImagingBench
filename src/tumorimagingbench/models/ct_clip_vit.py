import torch
from transformer_maskgit import CTViT
from huggingface_hub import hf_hub_download

from .utils import get_transforms
from .base import BaseModel


class CTClipVitExtractor(BaseModel):
    def __init__(self):
        super().__init__()
        self.spacing_xy = 0.75
        self.spacing_z = 1.5
        self.shape_xy = 480
        self.shape_z = 240
        self.model = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=60,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8
        )
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.transforms = get_transforms(
            orient="SLP",
            scale_range=(-1000, 1000),
            spatial_size=(60, 60, 60),
            spacing=(1.5, 0.75, 0.75),
            clamp=(-1, 1)
        )

    def load(self, weights_path: str = None):
        """
        Load pretrained CT-CLIP weights.
        Download the weights using:
        huggingface-cli download --repo-type dataset ibrahimhamamci/CT-RATE models/CT-CLIP-Related/CT-CLIP_v2.pt
        """
        if weights_path is None:
            weights_path = hf_hub_download(
                repo_id="ibrahimhamamci/CT-RATE",
                repo_type="dataset",
                filename="models/CT-CLIP-Related/CT-CLIP_v2.pt",
            )
        ckpt = torch.load(weights_path)
        vit_state_dict = {k.replace('visual_transformer.', ''): v for k, v in ckpt.items() if k.startswith('visual_transformer.')}
        self.model.load_state_dict(vit_state_dict)
        self.model.eval()

    def preprocess(self, x):
        return self.transforms(x)

    def forward(self, x, avg=True):
        with torch.no_grad():
            out = self.model(x, return_encoded_tokens=True)
            out = out.permute(0, 4, 1, 2, 3) # Make it NCDHW where C is feature dim
            return torch.nn.functional.adaptive_avg_pool3d(out, 1).flatten(start_dim=1) if avg else out