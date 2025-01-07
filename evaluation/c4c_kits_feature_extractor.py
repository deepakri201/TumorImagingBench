import pandas as pd
from base_feature_extractor import extract_all_features, save_features

def get_split_data(split):
    """Get LUNA dataset split"""
    split_paths = {
        "train": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/c4c-kits/train.csv",
        "val": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/c4c-kits/val.csv",
        "test": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/c4c-kits/test.csv"
    }
    if split not in split_paths:
        raise ValueError(f"Invalid split: {split}")
    return pd.read_csv(split_paths[split])

def preprocess_row(row):
    """Preprocess a row from LUNA dataset"""
    return row.copy()

def extract_features():
    """Extract features for LUNA dataset"""
    features = extract_all_features(get_split_data, preprocess_row)
    save_features(features, 'features/lndb.pkl')

if __name__ == "__main__":
    extract_features()
