from torch import nn
import torch
from typing import Union, List
import monai


class SegResNetDSwEmbedding(monai.networks.nets.SegResNetDS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _forward(self, x: torch.Tensor) -> Union[None, torch.Tensor, List[torch.Tensor]]:
        if self.preprocess is not None:
            x = self.preprocess(x)

        if not self.is_valid_shape(x):
            raise ValueError(f"Input spatial dims {x.shape} must be divisible by {self.shape_factor()}")

        x_down = self.encoder(x)

        x_down.reverse()
        x = x_down.pop(0)

        # Embedding (output features of the encoder) is returned together with the outputs
        embedding = x.clone()

        if len(x_down) == 0:
            x_down = [torch.zeros(1, device=x.device, dtype=x.dtype)]

        outputs: list[torch.Tensor] = []

        i = 0
        for level in self.up_layers:
            x = level["upsample"](x)
            x += x_down.pop(0)
            x = level["blocks"](x)

            if len(self.up_layers) - i <= self.dsdepth:
                outputs.append(level["head"](x))
            i = i + 1

        outputs.reverse()

        # in eval() mode, always return a single final output
        if not self.training:
            return outputs[0]

        return embedding, outputs
