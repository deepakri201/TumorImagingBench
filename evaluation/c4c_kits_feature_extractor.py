import pandas as pd
from base_feature_extractor import extract_all_features, save_features


def get_split_data(split):
    """Get dataset split for C4C-KiTS."""
    split_paths = {
        "train": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/c4c-kits/data.csv",
    }
    if split not in split_paths:
        print(f"Warning: Invalid split '{split}' for C4C-KiTS dataset.")
        return pd.DataFrame()
    return pd.read_csv(split_paths[split])


def preprocess_row(row):
    """Preprocess a row from the dataset."""
    return row


def extract_features():
    """Extract features for the C4C-KiTS dataset."""
    features = extract_all_features(get_split_data, preprocess_row)
    save_features(features, 'features/c4c_kits.pkl')


if __name__ == "__main__":
    extract_features()
