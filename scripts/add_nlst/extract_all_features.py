# extract_all_features.py 
# 
# Deepa Krishnaswamy
# Brigham and Women's Hospital 
# November 2025 
################################################

###############
### Imports ###
###############

import os 
import sys 
import pandas as pd 
import numpy as np 

sys.path.append("/home/exouser/Documents/git/TumorImagingBench/src/tumorimagingbench/")
sys.path.append("/home/exouser/Documents/git/TumorImagingBench/src/tumorimagingbench/models")
sys.path.append("/home/exouser/Documents/git/TumorImagingBench/src/tumorimagingbench/evaluation")

from base_feature_extractor import extract_features_for_model, extract_all_features, save_features 
from models import CTFMExtractor, FMCIBExtractor, MerlinExtractor, ModelsGenExtractor, PASTAExtractor, SUPREMExtractor, VISTA3DExtractor, VocoExtractor
# These two have warnings: CTClipVitExtractor,  MedImageInsightExtractor

#####################
### Set filenames ### 
#####################

# # for de_type classification
# output_directory = "/home/exouser/Documents/TumorImagingBench/nlst_sybil_de_type_classification"

# for de_stage classification
output_directory = "/home/exouser/Documents/TumorImagingBench/nlst_sybil_de_stage_classification"

# Set directories and filenames 
output_feature_directory = os.path.join(output_directory,"features")
if not os.path.isdir(output_feature_directory):
    os.makedirs(output_feature_directory)
train_filename = os.path.join(output_directory, "train.csv")
val_filename = os.path.join(output_directory, "val.csv")
test_filename = os.path.join(output_directory, "test.csv")

#################
### Functions ###
#################

# For the histology classification 
def get_split_data_fn(split):
    """Get dataset split."""
    split_paths = {
        "train": train_filename,
        "val": val_filename,
        "test": test_filename
    }
    if split not in split_paths:
        raise ValueError(f"Invalid split: {split}")
    return pd.read_csv(split_paths[split])

def preprocess_row_fn(row):
    """Preprocess a row from the dataset."""
    return row

##################
### Processing ###
##################

model_classes = [CTFMExtractor, 
                 FMCIBExtractor, 
                 MerlinExtractor, 
                 ModelsGenExtractor, 
                 PASTAExtractor, 
                 SUPREMExtractor, 
                 VISTA3DExtractor, 
                 VocoExtractor] 
model_classes_names = ['CTFM', 'FMCIB', 'Merlin', 'ModelsGen', 'PASTA', 'SUPREME', 'VISTA3D', 'Voco']

for model_class, model_class_name in zip(model_classes, model_classes_names):
    print('Processing ' + str(model_class_name))
    features = extract_features_for_model(model_class, get_split_data_fn, preprocess_row_fn)
    output_filename = os.path.join(output_feature_directory, model_class_name + '_features.pkl')
    save_features(features, output_filename) 
