import pandas as pd
from base_feature_extractor import extract_all_features, save_features
from functools import partial


def get_split_data(split, train_csv, val_csv, test_csv):
    """Get dataset split."""
    split_paths = {"train": train_csv, "val": val_csv, "test": test_csv}
    if split not in split_paths:
        raise ValueError(f"Invalid split: {split}")
    return pd.read_csv(split_paths[split])


def preprocess_row(row):
    """Preprocess a row from the LNDb dataset."""
    return row


def extract_features(output_path, train_csv, val_csv, test_csv, model_names=None):
    """Extract features for the LNDb dataset."""
    features = extract_all_features(partial(get_split_data, train_csv=train_csv, val_csv=val_csv, test_csv=test_csv), preprocess_row, model_names=model_names)
    save_features(features, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features for the LNDb dataset")
    parser.add_argument("--output", type=str, default="features/lndb.pkl",
                        help="Path where to save the extracted features")
    parser.add_argument("--train-csv", type=str, default="/home/suraj/Repositories/TumorImagingBench/data/eval/lndb/train.csv", help="Path to training CSV")
    parser.add_argument("--val-csv", type=str, default="/home/suraj/Repositories/TumorImagingBench/data/eval/lndb/val.csv", help="Path to validation CSV")
    parser.add_argument("--test-csv", type=str, default="/home/suraj/Repositories/TumorImagingBench/data/eval/lndb/test.csv", help="Path to test CSV")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Specific models to extract")
    args = parser.parse_args()
    extract_features(args.output, args.train_csv, args.val_csv, args.test_csv, args.models)
