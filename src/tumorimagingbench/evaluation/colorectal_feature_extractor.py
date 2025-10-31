import pandas as pd
from base_feature_extractor import extract_all_features, save_features
from functools import partial


def get_split_data(split, data_csv_path):
    """Get dataset split."""
    # This dataset only has training data
    if split != "train":
        return None
    return pd.read_csv(data_csv_path)


def preprocess_row(row):
    """Preprocess a row from the dataset."""
    return row


def extract_features(output_path, data_csv_path, model_names=None):
    """Extract features for the colorectal liver metastases dataset."""
    features = extract_all_features(partial(get_split_data, data_csv_path=data_csv_path), preprocess_row, model_names=model_names)
    save_features(features, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features for the colorectal liver metastases dataset")
    parser.add_argument("--output", type=str, default="features/colorectal_liver_metastases.pkl",
                        help="Path where to save the extracted features")
    parser.add_argument("--data-csv", type=str, default="/home/suraj/Repositories/TumorImagingBench/data/eval/colorectal_liver_metastases/data.csv", help="Path to dataset CSV")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Specific models to extract")
    args = parser.parse_args()
    extract_features(args.output, args.data_csv, args.models)
