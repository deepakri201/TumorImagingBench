# verify_tumor_location.py 
# 
# This file uses code from FMCIB to plot and verify the 
# location of tumors. 
# 
# Deepa Krishnaswamy 
# Brigham and Women's Hospital 
# November 2025 
##################################################

import os 
import sys 
import pandas as pd 
from fmcib.visualization import visualize_seed_point

csv_filename = "/home/exouser/Documents/TumorImagingBench/nlst_sybil_de_stage_classification/train.csv"
df_for_csv = pd.read_csv(csv_filename)

index = 1

row = dict(df_for_csv.iloc[index])
visualize_seed_point(row)