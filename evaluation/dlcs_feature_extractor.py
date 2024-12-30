import pandas as pd
import os
from base_feature_extractor import extract_all_features, save_features

def get_split_data(split):
    """Get DLCS dataset split"""
    df = pd.read_csv("/mnt/data1/datasets/DukeLungNoduleDataset/DLCSD24_Annotations.csv")
    return df[df["benchmark_split"].str.startswith(split)]

def preprocess_row(row):
    """Preprocess a row from DLCS dataset"""
    row = row.copy()
    row["image_path"] = f'/mnt/data1/datasets/DukeLungNoduleDataset/{row["ct_nifti_file"]}'

    if os.path.exists(row["image_path"]):
        return row
    else:
        return None

def extract_features():
    """Extract features for DLCS dataset"""
    features = extract_all_features(get_split_data, preprocess_row)
    save_features(features, 'features/dlcs.pkl')

if __name__ == "__main__":
    extract_features()
