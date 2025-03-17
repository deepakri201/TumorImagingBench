#!/usr/bin/env python3
"""
Generate saliency maps for different models on medical imaging datasets
using occlusion sensitivity analysis.
"""
import sys
import os
import gc
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.nn.functional import cosine_similarity
from monai.visualize import OcclusionSensitivity

# Add parent directory to path for importing local modules
sys.path.append('../')
from models import (
    CTClipVitExtractor,
    CTFMExtractor,
    FMCIBExtractor,
    MedImageInsightExtractor,
    MerlinExtractor,
    ModelsGenExtractor,
    PASTAExtractor,
    SUPREMExtractor,
    VISTA3DExtractor,
    VocoExtractor
)


def get_model_dict():
    """Return dictionary mapping model names to model classes for feature extraction."""
    models = [
        FMCIBExtractor,
        CTFMExtractor,
        CTClipVitExtractor,
        PASTAExtractor,
        VISTA3DExtractor,
        VocoExtractor,
        SUPREMExtractor,
        MerlinExtractor,
        MedImageInsightExtractor,
        ModelsGenExtractor,
    ]
    return {model.__name__: model for model in models}


def get_dataset_paths():
    """Return dictionary mapping dataset names to their file paths."""
    return {
        "LUNA": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/luna16/luna16/train.csv",
        "DLCS": "/mnt/data1/datasets/DukeLungNoduleDataset/DLCSD24_Annotations.csv",
        "NSCLC_Radiomics": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/nsclc_radiomics/train_annotations.csv",
        "NSCLC_Radiogenomics": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/nsclc_radiogenomics/train_annotations.csv",
        "C4KC-KiTs": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/c4c-kits/data.csv",
        "ColRecMet": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/colorectal_liver_metastases/data.csv",
    }


def process_dataset(encoder, encoder_name, dataset_name, dataset_path, save_dir, device="cuda:0", num_samples=5):
    """
    Process a dataset with the given encoder to generate saliency maps.
    
    Args:
        encoder: The encoder model for feature extraction
        encoder_name: Name of the encoder
        dataset_name: Name of the dataset
        dataset_path: Path to the dataset CSV file
        save_dir: Directory to save the results
        device: Device to run the model on
        num_samples: Number of samples to process
    """
    print(f"Processing {encoder_name} on {dataset_name}")
    
    try:
        samples = pd.read_csv(dataset_path).iloc[:num_samples]
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return
    
    # Initialize OcclusionSensitivity
    occ_sens = OcclusionSensitivity(
        nn_module=encoder, 
        n_batch=12, 
        activate=False, 
        mask_size=12, 
        overlap=0.25
    )
    
    distance_maps = []
    images = []

    for idx, sample in samples.iterrows():
        try:
            # Special handling for DLCS dataset
            if dataset_name == "DLCS":
                sample["image_path"] = f'/mnt/data1/datasets/DukeLungNoduleDataset/{sample["ct_nifti_file"]}'

            # Preprocess and move to device
            x = encoder.preprocess(sample)
            x = x.unsqueeze(0).to(device)
            
            # Generate occlusion map
            occ_map, _ = occ_sens(x)
            images.append(x)

            # Calculate distance using cosine similarity
            encoder.eval()
            with torch.no_grad():
                base_embedding = encoder(x)
                distances = 1 - cosine_similarity(
                    base_embedding.flatten().unsqueeze(0), 
                    occ_map.view(base_embedding.shape[-1], -1).t(), 
                    dim=1
                ).squeeze()

                distances = distances.view(*occ_map.shape[2:])
                distance_maps.append(distances.cpu())

                # Clean up memory
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"Error processing sample {idx} for {dataset_name}: {e}")
            continue

    # Save results if we have any
    if distance_maps:
        save_path = os.path.join(save_dir, f"saliency_{encoder_name}_{dataset_name}.torch")
        try:
            torch.save(
                {"distance_maps": distance_maps, "original_images": images},
                save_path,
            )
            print(f"Saved results to {save_path}")
        except Exception as e:
            print(f"Error saving results for {encoder_name} on {dataset_name}: {e}")


def main():
    """Main function to parse arguments and run the saliency map generation."""
    parser = argparse.ArgumentParser(
        description="Generate saliency maps for medical imaging models."
    )
    
    parser.add_argument(
        "--save_dir", 
        type=str,
        default="./saliency_results",
        help="Directory to save saliency maps (default: ./saliency_results)"
    )
    
    parser.add_argument(
        "--device", 
        type=str,
        default="cuda:0",
        help="Device to run models on (default: cuda:0)"
    )
    
    parser.add_argument(
        "--num_samples", 
        type=int,
        default=5,
        help="Number of samples to process per dataset (default: 5)"
    )
    
    parser.add_argument(
        "--models", 
        nargs='+',
        help="Specific models to run (default: all models)"
    )
    
    parser.add_argument(
        "--datasets", 
        nargs='+',
        help="Specific datasets to process (default: all datasets)"
    )
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {save_dir.absolute()}")
    
    # Get models and datasets
    all_models = get_model_dict()
    all_datasets = get_dataset_paths()
    
    # Filter models if specified
    if args.models:
        models = {name: all_models[name] for name in args.models if name in all_models}
        if not models:
            print(f"No valid models found. Available models: {list(all_models.keys())}")
            return
    else:
        models = all_models
    
    # Filter datasets if specified
    if args.datasets:
        datasets = {name: all_datasets[name] for name in args.datasets if name in all_datasets}
        if not datasets:
            print(f"No valid datasets found. Available datasets: {list(all_datasets.keys())}")
            return
    else:
        datasets = all_datasets
    
    # Process each model and dataset
    for encoder_name, encoder_class in models.items():
        try:
            encoder = encoder_class()
            encoder.to(args.device)
            
            for dataset_name, dataset_path in datasets.items():
                process_dataset(
                    encoder=encoder,
                    encoder_name=encoder_name,
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    save_dir=str(save_dir),
                    device=args.device,
                    num_samples=args.num_samples
                )
                
        except Exception as e:
            print(f"Error processing model {encoder_name}: {e}")
            continue
        
        # Clean up after each model
        del encoder
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
