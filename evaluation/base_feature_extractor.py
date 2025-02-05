import sys
import torch
from tqdm import tqdm
import pickle
import multiprocessing

sys.path.append('..')
from models.ct_fm_extractor import CTFMExtractor
from models.fmcib_extractor import FMCIBExtractor
from models.vista3d_extractor import VISTA3DExtractor
from models.voco_extractor import VocoExtractor
from models.suprem_extractor import SUPREMExtractor
from models.merlin_extractor import MerlinExtractor
from models.medimageinsight_extractor import MedImageInsightExtractor
from models.modelsgen_extractor import ModelsGenExtractor
from models.pyramid_extractor import PyramidExtractorVar, PyramidExtractorNoVar


def get_model_list():
    """Return list of model classes to use for feature extraction."""
    return [
        FMCIBExtractor,
        CTFMExtractor,
        PyramidExtractorVar,
        PyramidExtractorNoVar,
        VISTA3DExtractor,
        VocoExtractor,
        SUPREMExtractor,
        MerlinExtractor,
        MedImageInsightExtractor,
        ModelsGenExtractor,
    ]


def extract_features_for_model(model_class, get_split_data_fn, preprocess_row_fn):
    """Extract features for a single model across all splits."""
    model = model_class()
    print(f"\nProcessing {model.__class__.__name__}")
    model.load()

    model_features = {}

    with torch.no_grad():
        for split in ["train", "val", "test"]:
            split_df = get_split_data_fn(split)
            if split_df is None:
                continue

            model_features[split] = []

            for _, row in tqdm(
                split_df.iterrows(), total=len(split_df)
            ):
                row = preprocess_row_fn(row)
                if row is None:
                    continue
                image = model.preprocess(row)
                image = image.unsqueeze(0)
                feature = model.forward(image)
                if isinstance(feature, torch.Tensor):
                    feature = feature.cpu().numpy()
                model_features[split].append({
                    "feature": feature,
                    "row": row
                })

    return model_features


def extract_all_features(get_split_data_fn, preprocess_row_fn):
    """Extract features using all models."""
    feature_dict = {}
    model_classes = get_model_list()

    with multiprocessing.Pool() as pool:
        results = pool.starmap(
            extract_features_for_model,
            [(model_class, get_split_data_fn, preprocess_row_fn) for model_class in model_classes]
        )

    for model_class, model_features in zip(model_classes, results):
        feature_dict[model_class.__name__] = model_features

    return feature_dict


def save_features(feature_dict, output_file):
    """Save extracted features to a file."""
    with open(output_file, 'wb') as f:
        pickle.dump(feature_dict, f)
    print(f"Features saved to {output_file}")


if __name__ == "__main__":
    import pandas as pd

    def get_split_data(split):
        """Get dataset split."""
        split_paths = {
            "train": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/luna16/luna16/train.csv",
            "val": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/luna16/luna16/val.csv",
            "test": "/home/suraj/Repositories/FM-extractors-radiomics/data/eval/luna16/luna16/test.csv"
        }
        if split not in split_paths:
            raise ValueError(f"Invalid split: {split}")
        return pd.read_csv(split_paths[split])[:10]

    def preprocess_row(row):
        """Preprocess a row from the dataset."""
        return row

    def extract_features():
        """Extract features for the dataset."""
        features = extract_all_features(get_split_data, preprocess_row)
        save_features(features, 'features/test.pkl')

    extract_features()
