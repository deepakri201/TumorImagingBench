import pandas as pd
import os
from base_feature_extractor import extract_all_features, save_features


def get_split_data(split):
    """Get DLCS dataset split."""
    df = pd.read_csv("/mnt/data1/datasets/DukeLungNoduleDataset/DLCSD24_Annotations.csv")
    return df[df["benchmark_split"].str.startswith(split)]


def preprocess_row(row):
    """Preprocess a row from DLCS dataset."""
    row["image_path"] = f'/mnt/data1/datasets/DukeLungNoduleDataset/{row["ct_nifti_file"]}'

    if os.path.exists(row["image_path"]):
        return row
    else:
        return None


def extract_features(output_path):
    """Extract features for the DLCS dataset."""
    features = extract_all_features(get_split_data, preprocess_row)
    save_features(features, output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features for the DLCS dataset")
    parser.add_argument("--output", type=str, default="features/dlcs.pkl", 
                        help="Path where to save the extracted features")
    args = parser.parse_args()
    extract_features(args.output)
