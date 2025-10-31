import pandas as pd
from base_feature_extractor import extract_all_features, save_features
from functools import partial

def get_split_data(split, train_csv, val_csv, test_csv):
    """Get dataset split."""
    split_paths = {
        "train": train_csv,
        "val": val_csv,
        "test": test_csv
    }
    if split not in split_paths:
        raise ValueError(f"Invalid split: {split}")
    return pd.read_csv(split_paths[split])


def preprocess_row(row):
    """Preprocess a row from the NSCLC-Radiogenomics dataset."""
    return row


def extract_features(output_path, train_csv, val_csv, test_csv):
    """Extract features for the NSCLC-Radiogenomics dataset."""

    features = extract_all_features(partial(get_split_data, train_csv=train_csv, val_csv=val_csv, test_csv=test_csv), preprocess_row)
    save_features(features, output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features for the NSCLC-Radiogenomics dataset")
    parser.add_argument("--output", type=str, default="features/nsclc_radiogenomics.pkl",
                        help="Path where to save the extracted features")
    parser.add_argument("--train-csv", type=str, default="/home/suraj/Repositories/TumorImagingBench/data/eval/nsclc_radiogenomics/train_annotations.csv", help="Path to training CSV")
    parser.add_argument("--val-csv", type=str, default="/home/suraj/Repositories/TumorImagingBench/data/eval/nsclc_radiogenomics/val_annotations.csv", help="Path to validation CSV")
    parser.add_argument("--test-csv", type=str, default="/home/suraj/Repositories/TumorImagingBench/data/eval/nsclc_radiogenomics/test_annotations.csv", help="Path to test CSV")
    args = parser.parse_args()
    extract_features(args.output, args.train_csv, args.val_csv, args.test_csv)
