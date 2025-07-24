import pandas as pd
from base_feature_extractor import extract_all_features, save_features


def get_split_data(split):
    """Get dataset split for colorectal liver metastases."""
    split_paths = {
        "train": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/colorectal_liver_metastases/data.csv",
    }
    if split not in split_paths:
        print(f"Warning: Invalid split '{split}' for colorectal liver metastases dataset.")
        return None
    return pd.read_csv(split_paths[split])


def preprocess_row(row):
    """Preprocess a row from the dataset."""
    return row


def extract_features(output_path):
    """Extract features for the colorectal liver metastases dataset."""
    features = extract_all_features(get_split_data, preprocess_row)
    save_features(features, output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features for the C4C-KiTS dataset")
    parser.add_argument("--output", type=str, default="features/colorectal_liver_metastases.pkl", 
                        help="Path where to save the extracted features")
    args = parser.parse_args()
    extract_features(args.output)
