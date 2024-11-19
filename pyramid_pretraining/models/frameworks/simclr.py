import torch
import torch.nn as nn
from lightly.models.modules import SimCLRProjectionHead


class SimCLR(nn.Module):
    """
    This class implements the SimCLR model.

    Attributes:
        backbone (nn.Module): The backbone network used in the SimCLR model.
        num_ftrs (int): The number of output features from the backbone network. Default is 32.
        out_dim (int): The dimension of the output representations. Default is 128.
        projection_head (SimCLRProjectionHead): The projection head used in the SimCLR model.
    """

    def __init__(self, backbone: nn.Module, num_ftrs: int = 32, out_dim: int = 128, spatial_dims: int = 3):
        """
        Constructs the SimCLR model with a given backbone network, number of features, and output dimension.

        Args:
            backbone (nn.Module): The backbone network.
            num_ftrs (int, optional): The number of features. Default is 32.
            out_dim (int, optional): The output dimension. Default is 128.
        """
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(num_ftrs, num_ftrs // 2, out_dim, batch_norm=False)
        if spatial_dims == 2:
            self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif spatial_dims == 3:
            self.average_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        else:
            raise ValueError("`spatial_dims` must be 2 or 3.")

    def forward(self, input):
        """
        Defines the computation performed at every call.

        Args:
            input (torch.Tensor or tuple or list): The input data. It can be a tensor, a tuple, or a list. Nested 
                structures are also supported.

        Returns:
            torch.Tensor or list: The output of the forward pass. If the input is a tensor, a tensor is returned.
                If the input is a tuple or a list, a list is returned.
        """
        def tensor_forward(x):
            assert isinstance(x, torch.Tensor)
            x = self.backbone(x)
            x = self.average_pool(x).flatten(start_dim=1)
            x = self.projection_head(x)
            return x

        def sequence_forward(x):
            assert isinstance(x, tuple) or isinstance(x, list)
            out = []
            for sample in x:
                if isinstance(sample, torch.Tensor):
                    out.append(tensor_forward(sample))
                elif isinstance(sample, tuple) or isinstance(sample, list):
                    out.append(sequence_forward(sample))
            return out

        if isinstance(input, torch.Tensor):
            return tensor_forward(input)

        if isinstance(input, tuple) or isinstance(input, list):
            return sequence_forward(input)
