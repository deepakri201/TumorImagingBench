import sys
import torch
from tqdm import tqdm
import pickle

sys.path.append('..')
from models.ct_fm_extractor import CTFMExtractor
from models.fmcib_extractor import FMCIBExtractor
from models.vista3d_extractor import VISTA3DExtractor
from models.voco_extractor import VocoExtractor
from models.suprem_extractor import SUPREMExtractor
from models.merlin_extractor import MerlinExtractor
from models.medimageinsight_extractor import MedImageInsightExtractor

def get_model_list():
    """Return list of model classes to use for feature extraction"""
    return [
       FMCIBExtractor
    ]

def extract_features_for_model(model_class, get_split_data_fn, preprocess_row_fn):
    """Extract features for a single model across all splits"""
    model = model_class()
    print(f"\nProcessing {model.__class__.__name__}")
    model.load()
    
    model_features = {}
    
    with torch.no_grad():
        for split in ["train", "val", "test"]:
            split_df = get_split_data_fn(split)
            model_features[split] = []
            for _, row in tqdm(split_df.iterrows(), total=len(split_df),
                             desc=f"Processing {split} set",
                             position=2, leave=False):
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
    """Extract features using all models"""
    feature_dict = {}
    
    for model_class in tqdm(get_model_list(), desc="Overall Progress", position=0):
        feature_dict[model_class.__name__] = extract_features_for_model(
            model_class, get_split_data_fn, preprocess_row_fn
        )
    
    return feature_dict

def save_features(feature_dict, output_file):
    """Save extracted features to file"""
    with open(output_file, 'wb') as f:
        pickle.dump(feature_dict, f)
    print(f"Features saved to {output_file}")
    print(feature_dict)
