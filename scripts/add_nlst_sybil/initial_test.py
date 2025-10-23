# initial_test.py 
# 
# Here we assume that we already have a csv with the nifti file/point location/ground truth 
# 


import tumorimagingbench as tib
tib.check_module_status()

import pandas as pd
import sys
sys.path.append("/Users/dk422/git/TumorImagingBench/src/tumorimagingbench/")
sys.path.append("/Users/dk422/git/TumorImagingBench/src/tumorimagingbench/models")
sys.path.append("/Users/dk422/git/TumorImagingBench/src/tumorimagingbench/evaluation")

input_path = "/Users/dk422/Documents/Projects/TumorImagingBench/fmcib_input1.csv"

from base_feature_extractor import extract_features_for_model, extract_all_features, save_features 
from models import CTFMExtractor, FMCIBExtractor, MerlinExtractor, ModelsGenExtractor, PASTAExtractor, SUPREMExtractor, VISTA3DExtractor, VocoExtractor
# These two have warnings: CTClipVitExtractor,  MedImageInsightExtractor

def get_split_data_fn(split):
    """Get dataset split."""
    split_paths = {
        "test": input_path
    }
    if split not in split_paths:
        raise ValueError(f"Invalid split: {split}")
    return pd.read_csv(split_paths[split])[:10]

def preprocess_row_fn(row):
    """Preprocess a row from the dataset."""
    return row

# Not sure why this isn't working
# features = extract_all_features(get_split_data_fn, preprocess_row_fn)

# This one is giving an assertion error saying "Torch not compiled with CUDA enabled"
# model_class = FMCIBExtractor 

model_class = CTFMExtractor
features = extract_features_for_model(model_class, get_split_data_fn, preprocess_row_fn)
save_features(features, '/Users/dk422/Documents/Projects/TumorImagingBench/features_fmcib.pkl')

# import tumorimagingbench as tib
# tib.check_module_status()
# # tib.models.AVAILABLE_EXTRACTORS

# from fmcib.datasets import generate_dummy_data
# import pandas as pd

# import sys
# sys.path.append("/Users/dk422/git/TumorImagingBench/src/tumorimagingbench/")
# sys.path.append("/Users/dk422/git/TumorImagingBench/src/tumorimagingbench/models")
# sys.path.append("/Users/dk422/git/TumorImagingBench/src/tumorimagingbench/evaluation")

# from models import FMCIBExtractor

# generate_dummy_data("data", size=1)
# sample_path = "/Users/dk422/git/TumorImagingBench/data/dummy.csv"
# model = FMCIBExtractor()
# model.load()
# # features = model.extract(sample_path)

# # from base_feature_extractor import extract_all_features
# # from base_feature_extractor import extract_features_for_model

