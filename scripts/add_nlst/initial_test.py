import pandas as pd
import os 
import sys
sys.path.append("/home/exouser/Documents/git/TumorImagingBench/src/tumorimagingbench/")
sys.path.append("/home/exouser/Documents/git/TumorImagingBench/src/tumorimagingbench/models")
sys.path.append("/home/exouser/Documents/git/TumorImagingBench/src/tumorimagingbench/evaluation")

import tumorimagingbench as tib
tib.check_module_status()

input_path = "/home/exouser/Documents/TumorImagingBench/fmcib_input1.csv"

from base_feature_extractor import extract_features_for_model, extract_all_features, save_features 
from models import CTFMExtractor, FMCIBExtractor, MerlinExtractor, ModelsGenExtractor, PASTAExtractor, SUPREMExtractor, VISTA3DExtractor, VocoExtractor
# These two have warnings: CTClipVitExtractor,  MedImageInsightExtractor

def get_split_data_fn(split):
    """Get dataset split."""
    split_paths = {
        "train": input_path,
        "val": input_path,
        "test": input_path
    }
    if split not in split_paths:
        raise ValueError(f"Invalid split: {split}")
    return pd.read_csv(split_paths[split])

def preprocess_row_fn(row):
    """Preprocess a row from the dataset."""
    return row

# These are working 
# model_classes = [CTFMExtractor, FMCIBExtractor, MerlinExtractor, ModelsGenExtractor, PASTAExtractor, SUPREMExtractor, VISTA3DExtractor, VocoExtractor]
# model_classes_names = ['CTFM', 'FMCIB', 'Merlin', 'ModelsGen', 'SUPREME', 'VISTA3D']

# These are not working
# model_classes = [PASTAExtractor, VocoExtractor]
# model_classes_names = ['PASTA', 'Voco']

# model_classes = [PASTAExtractor]
# model_classes_names = ['PASTA']

model_classes = [VocoExtractor]
model_classes_names = ['Voco']

for model_class, model_class_name in zip(model_classes, model_classes_names):
    print('Processing ' + str(model_class_name))
    features = extract_features_for_model(model_class, get_split_data_fn, preprocess_row_fn)
    output_filename = os.path.join('/home/exouser/Documents/TumorImagingBench/features_test', model_class_name + '_features_test.pkl')
    save_features(features, output_filename) 
    # try:
    #     features = extract_features_for_model(model_class, get_split_data_fn, preprocess_row_fn)
    #     output_filename = os.path.join('/home/exouser/Documents/TumorImagingBench/features_test', model_class_name + '_features_test.pkl')
    #     save_features(features, output_filename) 
    # except:
    #     print('Unable to extract features for: ' + str(model_class_name))
