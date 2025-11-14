# create_csv_incorrect_tumors.py 
# 
# There were some issues with the extraction of crops from the NLST data. 
# Therefore our tumors were incorrect. Therefore I manually identified which
# ones were wrong, due to things like: 
#    - flipped orientation in z direction
#    - multiple volumes in a single series 
#    - incorrect reconstruction of DICOM slices into nifti 
#    - small number of transverse slices in series 
# 
# Deepa Krishnaswamy 
# Brigham and Women's Hospital 
# November 2025 
########################################################

import os 
import sys 
import pandas as pd 
import numpy as np 
from pathlib import Path

# The original csv file 
original_csv_filename = "/home/exouser/Documents/TumorImagingBench/nlst_sybil_analysis/nlst_sybil_updated_paths.csv"
# The directory of the pngs with incorrect tumor locations 
tumor_png_directory = "/home/exouser/Documents/TumorImagingBench/nlst_sybil_analysis/verify_tumor_location_incorrect"
# The output csv files 
without_incorrect_tumor_csv_filename = "/home/exouser/Documents/TumorImagingBench/nlst_sybil_analysis/nlst_sybil_updated_paths_without_incorrect_tumors.csv"
only_incorrect_tumor_csv_filename = "/home/exouser/Documents/TumorImagingBench/nlst_sybil_analysis/nlst_sybil_updated_paths_only_incorrect_tumors.csv"

original_df = pd.read_csv(original_csv_filename)
incorrect_tumor_sop = [f for f in os.listdir(tumor_png_directory) if f.endswith('.png')]
incorrect_tumor_sop = [Path(f).stem for f in incorrect_tumor_sop]

new_df = original_df.copy(deep=True)
new_df = new_df[~new_df['SOPInstanceUID'].isin(incorrect_tumor_sop)] # or remove the full patient? 
new_df.to_csv(without_incorrect_tumor_csv_filename)

temp_df = original_df[original_df['SOPInstanceUID'].isin(incorrect_tumor_sop)]
temp_df.to_csv(only_incorrect_tumor_csv_filename)

num_patients_orig = len(sorted(list(set(original_df['PatientID'].values))))
num_series_orig = len(sorted(list(set(original_df['SeriesInstanceUID'].values))))
num_sop_orig = len(sorted(list(set(original_df['SOPInstanceUID'].values))))

num_patients_new = len(sorted(list(set(new_df['PatientID'].values))))
num_series_new = len(sorted(list(set(new_df['SeriesInstanceUID'].values))))
num_sop_new = len(sorted(list(new_df['SOPInstanceUID'].values)))

print('len original: ' + str(len(original_df)))
print('num_patients_orig: ' + str(num_patients_orig))
print('num_series_orig: ' + str(num_series_orig))
print('num_sop_orig: ' + str(num_sop_orig))

print('fixed df: ' + str(len(new_df)))
print('num_patients_new: ' + str(num_patients_new))
print('num_series_new: ' + str(num_series_new))
print('num_sop_new: ' + str(num_sop_new))


