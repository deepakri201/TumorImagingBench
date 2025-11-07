# create_results_figures.py 
# 
# Deepa Krishnaswamy 
# Brigham and Women's Hospital 
# November 2025
#########################################

###############
### Imports ###
###############

import os 
import sys 
import numpy as np 
import pandas as pd 

from pathlib import Path
import plotly.express as px

##################
### Set files ###
##################

# de type classification 
metrics_directory = "/home/exouser/Documents/TumorImagingBench/nlst_sybil_de_type_classification/metrics" 

# de stag classisfication 

# Set directories and filenames 
scores_directory = os.path.join(metrics_directory, "scores")
CTFM_filename = os.path.join(scores_directory, "CTFM_features_test.npz")
FMCIB_filename = os.path.join(scores_directory, "FMCIB_features_test.npz")
Merlin_filename = os.path.join(scores_directory, "Merlin_features_test.npz")
ModelsGen_filename = os.path.join(scores_directory, "ModelsGen_features_test.npz")
PASTA_filename = os.path.join(scores_directory, "PASTA_features_test.npz")
SUPREME_filename = os.path.join(scores_directory, "SUPREME_features_test.npz")
VISTA3D_filename = os.path.join(scores_directory, "VISTA3D_features_test.npz")
Voco_filename = os.path.join(scores_directory, "Voco_features_test.npz")

output_png_filename = os.path.join(metrics_directory, 'accuracy_over_models.png')

#################
### Load data ### 
#################

scores_filename_list =   [CTFM_filename, 
                          FMCIB_filename, 
                          Merlin_filename,
                          ModelsGen_filename,
                          PASTA_filename,
                          SUPREME_filename,
                          VISTA3D_filename,
                          Voco_filename]
fm_models = [os.path.basename(f) for f in scores_filename_list] 
fm_models = [Path(f).stem for f in fm_models]
fm_models = [f.split('_')[0] for f in fm_models]

scores = [] 
for scores_filename in scores_filename_list: 
    data = np.load(scores_filename)['score']
    scores.append(data)

df = pd.DataFrame()
df['FM'] = fm_models 
df['test_accuracy'] = scores 

#######################
### Create auc plot ###
#######################

fig = px.bar(df, x='FM', y='test_accuracy', title='Test accuracy per foundation model')
fig.write_image(output_png_filename)