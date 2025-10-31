import pandas as pd
from base_feature_extractor import extract_all_features, save_features
from functools import partial


def get_split_data(split, data_csv_path):
    """Get dataset split."""
    # Returns same data for all splits (used for test-retest analysis)
    return pd.read_csv(data_csv_path)


def preprocess_row(row):
    """Preprocess a row from the NSCLC-Radiomics dataset."""
    return row


def extract_features(output_path, data_csv_path, model_names=None):
    """Extract features for the NSCLC-Radiomics stability dataset."""
    features = extract_all_features(partial(get_split_data, data_csv_path=data_csv_path), preprocess_row, model_names=model_names)
    save_features(features, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features for the NSCLC-Radiomics stability dataset")
    parser.add_argument("--output", type=str, default="features/nsclc_radiomics_stability.pkl",
                        help="Path where to save the extracted features")
    parser.add_argument("--data-csv", type=str, default="/home/suraj/Repositories/TumorImagingBench/data/eval/nsclc_radiomics/stability.csv", help="Path to stability dataset CSV")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Specific models to extract")
    args = parser.parse_args()
    extract_features(args.output, args.data_csv, args.models)
