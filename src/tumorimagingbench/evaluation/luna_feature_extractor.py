import pandas as pd
from base_feature_extractor import extract_all_features, save_features


def get_split_data(split):
    """Get LUNA dataset split."""
    split_paths = {
        "train": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/luna16/luna16/train.csv",
        "val": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/luna16/luna16/val.csv",
        "test": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/luna16/luna16/test.csv"
    }
    if split not in split_paths:
        raise ValueError(f"Invalid split: {split}")
    return pd.read_csv(split_paths[split])


def preprocess_row(row):
    """Preprocess a row from the LUNA dataset."""
    return row


def extract_features(output_path):
    """Extract features for the LUNA dataset."""
    features = extract_all_features(get_split_data, preprocess_row)
    save_features(features, output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features for the LUNA dataset")
    parser.add_argument("--output", type=str, default="features/luna.pkl", 
                        help="Path where to save the extracted features")
    args = parser.parse_args()
    extract_features(args.output)
