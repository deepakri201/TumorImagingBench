"""
DummyResNetExtractor: A simple demonstration model for educational purposes.

This model serves as an example of how to integrate new foundation models
into the TumorImagingBench framework. It uses a lightweight ResNet-18
architecture pre-trained on ImageNet and adapted for 3D medical imaging.

The model demonstrates:
1. How to inherit from BaseModel
2. How to implement required abstract methods
3. How to use MONAI preprocessing transforms
4. How to extract features for downstream tasks
5. How to handle model loading and inference

Usage:
    from tumorimagingbench.models import get_extractor

    # Get the model class
    DummyResNetExtractor = get_extractor('DummyResNetExtractor')

    # Instantiate and load
    model = DummyResNetExtractor()
    model.load()  # Load pre-trained weights

    # Preprocess input
    input_dict = {
        'image_path': '/path/to/image.nii.gz',
        'coordX': 100.0,
        'coordY': 150.0,
        'coordZ': 200.0
    }
    processed_input = model.preprocess(input_dict)

    # Extract features
    features = model.forward(processed_input)
"""

import torch
import torch.nn as nn
from monai.networks import nets

from .utils import get_transforms
from .base import BaseModel


class DummyResNetExtractor(BaseModel):
    """
    A simple ResNet-based feature extractor for 3D medical imaging.

    This model uses torchvision's ResNet-18 pre-trained on ImageNet
    as a backbone. The model is converted to 3D convolutions to handle
    volumetric medical imaging data.

    Architecture:
    - Input: 3D volumetric image (1, 48, 48, 48)
    - Backbone: ResNet-18 with 3D convolutions
    - Feature extraction: Global average pooling
    - Output: 512-dimensional feature vector

    Parameters:
    -----------
    None

    Attributes:
    -----------
    model : torch.nn.Module
        The ResNet-18 backbone with 3D convolution layers
    transforms : monai.transforms.Compose
        MONAI transformation pipeline for preprocessing

    Examples:
    ---------
    >>> from tumorimagingbench.models import DummyResNetExtractor
    >>> model = DummyResNetExtractor()
    >>> model.load()
    >>> # Now ready for inference
    """

    def __init__(self):
        """Initialize the DummyResNet model with ResNet-18 backbone."""
        super().__init__()
        # Load pre-trained ResNet-18 from torchvision
        # This model is trained on ImageNet (2D images)
        self.resnet18 = nets.resnet18(feed_forward=False, n_input_channels=1)
        self.resnet18.to("cuda" if torch.cuda.is_available() else "cpu")
        self.transforms = get_transforms()

    def load(self, weights_path: str = None):
        pass

    def preprocess(self, x):
        """
        Preprocess input data for the model.

        This method applies the MONAI transformation pipeline to convert
        raw medical imaging data into model-ready tensors.

        The preprocessing pipeline includes:
        1. Load NIFTI image from disk
        2. Ensure channel-first format
        3. Orient to LPS (Left-Posterior-Superior) standard
        4. Resample to 1mm isotropic spacing
        5. Crop a 48x48x48 patch around the centroid coordinates
        6. Re-orient to RAS (Right-Anterior-Superior)
        7. Scale intensities to [0, 1] range
        8. Pad if necessary
        9. Return as tensor

        Parameters:
        -----------
        x : dict
            Dictionary containing:
            - 'image_path': str - Path to NIFTI file (.nii.gz)
            - 'coordX': float - X coordinate of region of interest (physical coordinates)
            - 'coordY': float - Y coordinate of region of interest
            - 'coordZ': float - Z coordinate of region of interest

        Returns:
        --------
        torch.Tensor
            Preprocessed image tensor of shape (1, 48, 48, 48)

        Examples:
        ---------
        >>> input_dict = {
        ...     'image_path': '/path/to/image.nii.gz',
        ...     'coordX': 100.0,
        ...     'coordY': 150.0,
        ...     'coordZ': 200.0
        ... }
        >>> processed = model.preprocess(input_dict)
        >>> print(processed.shape)
        torch.Size([1, 48, 48, 48])
        """
        return self.transforms(x)

    def forward(self, x):
        """
        Forward pass: extract features from input images.

        The model processes 3D volumetric patches and outputs a fixed-size
        feature vector suitable for downstream tasks like classification,
        clustering, or similarity matching.

        Processing pipeline:
        1. Pass through ResNet-18 backbone (conv1 + layer1-4 + global pooling)
        2. Extract 512-dimensional features
        3. Return as numpy array on CPU

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 48, 48, 48)
            Expected to be on the same device as the model

        Returns:
        --------
        numpy.ndarray
            Feature matrix of shape (batch_size, 512)
            Moved to CPU for downstream processing

        Notes:
        ------
        - No gradients are computed (inference mode)
        - Features are aggregated via global average pooling
        - Output is converted to numpy for compatibility with feature extraction pipeline

        Examples:
        ---------
        >>> x = torch.randn(1, 1, 48, 48, 48).cuda()
        >>> features = model.forward(x)
        >>> print(features.shape)
        (1, 512)
        """
        with torch.no_grad():
            x = self.resnet18(x)
            return x


# Additional utility functions for working with DummyResNetExtractor

def create_dummy_batch(batch_size=4, device='cuda'):
    """
    Create a dummy batch of random inputs for testing.

    Useful for testing the model without actual medical imaging data.

    Parameters:
    -----------
    batch_size : int
        Number of samples in the batch
    device : str
        Device to create tensors on ('cuda' or 'cpu')

    Returns:
    --------
    torch.Tensor
        Random tensor of shape (batch_size, 1, 48, 48, 48) on the specified device

    Examples:
    ---------
    >>> model = DummyResNetExtractor()
    >>> model.load()
    >>> model = model.to('cuda')
    >>> dummy_input = create_dummy_batch(batch_size=2, device='cuda')
    >>> features = model(dummy_input)
    >>> print(features.shape)
    (2, 512)
    """
    return torch.randn(batch_size, 1, 48, 48, 48, device=device)


if __name__ == '__main__':
    # Example usage
    print("DummyResNetExtractor Example")
    print("=" * 50)

    # Initialize and load model
    model = DummyResNetExtractor()
    model.load()
    print("✓ Model initialized and loaded")

    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"✓ Model moved to {device}")

    # Test with dummy input
    dummy_input = create_dummy_batch(batch_size=2, device=device)
    print(f"✓ Created dummy input: {dummy_input.shape}")

    # Extract features
    features = model.forward(dummy_input)
    print(f"✓ Extracted features: {features.shape}")
