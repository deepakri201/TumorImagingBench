import argparse
import torch
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the models
sys.path.append("..")

import models

def get_model(model_name):
    if hasattr(models, model_name):
        return getattr(models, model_name)()
    else:
        raise ValueError(f"{model_name} not found")

def main(args):

    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a dummy 3D volume simulating a CT scan
    import SimpleITK as sitk
    dummy_volume = np.random.randint(-1024, 3000, size=(300, 300, 300)).astype(np.float32)
    sitk_img = sitk.GetImageFromArray(dummy_volume)
    sitk_img.SetOrigin((0.0, 0.0, 0.0))
    sitk_img.SetSpacing((1.0, 1.0, 1.0))
    tmp_dir = "/tmp"
    dummy_file = os.path.join(tmp_dir, "dummy_volume.nii.gz")
    sitk.WriteImage(sitk_img, dummy_file)
    print(f"Dummy NIFTI file saved to: {dummy_file}")
    
    # Compute centroid in physical space using SimpleITK parameters
    size = np.array(sitk_img.GetSize(), dtype=np.float32)
    spacing = np.array(sitk_img.GetSpacing(), dtype=np.float32)
    origin = np.array(sitk_img.GetOrigin(), dtype=np.float32)
    centroid = origin + spacing * ((size - 1) / 2.0)
    coordX, coordY, coordZ = centroid
    print(f"Centroid coordinates (physical space): coordX={coordX}, coordY={coordY}, coordZ={coordZ}")
    
    # Create model based on the provided command-line argument
    model = get_model(args.model)
    
    # Load model weights (will download from HuggingFace if no path provided)
    print("Loading model weights...")
    model.load()
    model.model.to(device)
    print("Model loaded successfully!")
    
    # Preprocess the input by providing a dictionary with the file path and centroid coordinates
    print("Preprocessing input...")
    input_dict = {
        "image_path": dummy_file,
        "coordX": coordX,
        "coordY": coordY,
        "coordZ": coordZ
    }

    total_params = sum(p.numel() for p in model.model.parameters())
    total_params_million = total_params / 1e6
    print(f"Model parameter count: {total_params_million:.2f}M")
    preprocessed_input = model.preprocess(input_dict)
    preprocessed_input = preprocessed_input.unsqueeze(0).to(device)
    print(f"Preprocessed input shape: {preprocessed_input.shape}")
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model.forward(preprocessed_input)
    
    # Print output information
    if isinstance(output, (tuple, list)):
        print(f"Output is a tuple/list with {len(output)} elements")
        for i, out in enumerate(output):
            print(f"  Output[{i}] shape: {out.shape}")
    else:
        print(f"Output shape: {output.shape}")
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a dummy volume with the selected model")
    parser.add_argument("--model", type=str, default="PASTAExtractor", help="Name of the model to use (e.g., 'PASTAExtractor')")
    args = parser.parse_args()
    main(args)
