import pandas as pd
from base_feature_extractor import extract_all_features, save_features


def get_split_data(split):
    return pd.read_csv("/home/suraj/Repositories/FM-extractors-radiomics/data/eval/nsclc_radiomics/stability.csv")


def preprocess_row(row):
    """Preprocess a row from the NSCLC-Radiomics dataset."""
    return row


def extract_features(output_path):
    """Extract features for the NSCLC-Radiomics dataset."""
    features = extract_all_features(get_split_data, preprocess_row)
    save_features(features, output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features for the NSCLC-Radiomics dataset")
    parser.add_argument("--output", type=str, default="features/nsclc_radiomics_stability.pkl", 
                        help="Path where to save the extracted features")
    args = parser.parse_args()
    extract_features(args.output)
