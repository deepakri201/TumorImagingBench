# create_tumor_csv_file.py 
# 
# This script creates a csv file that can be used by each of the 
# extractors to obtain the features. 
# - We modify the file paths to match the nifti_directory 
# - We only keep certain labels (de_stag_mapped, de_type_mapped), and 
#   create a new column called label 
# 
# Deepa Krishnaswamy 
# Brigham and Women's Hospital 
# November 2025
############################################################

###############
### Imports ###
############### 

import os 
import sys
import numpy as np  
import pandas as pd 

##############
### Inputs ### 
##############

# # De type classification 
# col_type = "labels_de_type_mapped" 
# main_csv_filename = "/home/exouser/Documents/TumorImagingBench/nlst_sybil.csv"
# output_csv_directory = "/home/exouser/Documents/TumorImagingBench/nlst_sybil_de_type_classification"

# De stage classification 
col_type = "labels_de_stag_mapped" 
main_csv_filename = "/home/exouser/Documents/TumorImagingBench/nlst_sybil.csv"
output_csv_directory = "/home/exouser/Documents/TumorImagingBench/nlst_sybil_de_stage_classification"

# Set other filenames 
output_csv_filename = os.path.join(output_csv_directory, "main.csv")
output_csv_filename_train = os.path.join(output_csv_directory, "train.csv")
output_csv_filename_val = os.path.join(output_csv_directory, "val.csv")
output_csv_filename_test = os.path.join(output_csv_directory, "test.csv")
nifti_directory = "/home/exouser/Documents/TumorImagingBench/nlst_data/nifti" 
train_size = 0.6
val_size = 0.2
test_size = 0.2

if not os.path.isdir(output_csv_directory):
    os.makedirs(output_csv_directory,exist_ok=True)

##################
### Processing ### 
##################

# Load dataframe 
df_main = pd.read_csv(main_csv_filename)
df_output = df_main.copy(deep=True) 

# Create new image paths 
image_paths = df_output['image_path'].values
image_paths_filenames = [os.path.basename(f) for f in image_paths]
image_paths_new = [os.path.join(nifti_directory,f) for f in image_paths_filenames]
df_output['image_path'] = image_paths_new 

# Set the labels depending on the col_type 
if (col_type=="labels_de_type_mapped"): 
    labels_map = {
                  "Adenocarcinoma, NOS": 0, 
                  "Squamous cell carcinoma, NOS": 1
                  }
elif (col_type=="labels_de_stag_mapped"): 
    labels_map = {
                  0: 0, # stage IA
                  1: 0, # stage 1B
                  2: 1, # stage IIA
                  3: 1, # stage IIB
                  4: 1, # stage IIIA
                  5: 1, # stage IIB
                  6: 1  # stage IV 
                  }

# Keep only certain labels         
labels_keep = labels_map.keys()
df_output = df_output[df_output[col_type].isin(labels_keep)]
# Create a new "labels" column 
df_output["label"] = df_output[col_type].map(labels_map)
# Save as csv - backup  
df_output.to_csv(output_csv_filename)

# Now divide into train, val and test, and save the csv files 
# Make sure to split by patient
patients = sorted(list(set(df_output['PatientID'].values)))
num_patients = len(patients)
num_train_patients = np.int32(np.floor(num_patients * train_size))
num_val_patients = np.int32(np.floor(num_patients * val_size))
train_patients = patients[0:num_train_patients]
val_patients = patients[num_train_patients:num_train_patients+num_val_patients]
test_patients = patients[num_train_patients+num_val_patients::]
# Create the dataframes 
df_train = df_output[df_output['PatientID'].isin(train_patients)]
df_val = df_output[df_output['PatientID'].isin(val_patients)]
df_test = df_output[df_output['PatientID'].isin(test_patients)]
df_train.to_csv(output_csv_filename_train)
df_val.to_csv(output_csv_filename_val)
df_test.to_csv(output_csv_filename_test)

print('num_train_patients: ' + str(len(train_patients)))
print('num_val_patients: ' + str(len(val_patients)))
print('num_test_patients: ' + str(len(test_patients)))

print('Train size: ' + str(len(df_train)))
print('Val size: ' + str(len(df_val)))
print('Test size: ' + str(len(df_test)))





