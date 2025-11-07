# train_and_eval_classifiers.py 
# 
# Deepa Krishnaswamy 
# Brigham and Women's Hospital 
# November 2025 
###############################################################

###############
### Imports ### 
###############

import os 
import sys 
import numpy as np 
import pandas as pd 
import pickle 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt 

from pathlib import Path

###############
### Loading ###
###############

features_directory = "/home/exouser/Documents/TumorImagingBench/nlst_sybil_de_type_classification/features"
CTFM_filename = os.path.join(features_directory, "CTFM_features_test.pkl")
FMCIB_filename = os.path.join(features_directory, "FMCIB_features_test.pkl")
Merlin_filename = os.path.join(features_directory, "Merlin_features_test.pkl")
ModelsGen_filename = os.path.join(features_directory, "ModelsGen_features_test.pkl")
PASTA_filename = os.path.join(features_directory, "PASTA_features_test.pkl")
SUPREME_filename = os.path.join(features_directory, "SUPREME_features_test.pkl")
VISTA3D_filename = os.path.join(features_directory, "VISTA3D_features_test.pkl")
Voco_filename = os.path.join(features_directory, "Voco_features_test.pkl")

metrics_directory = "/home/exouser/Documents/TumorImagingBench/nlst_sybil_de_type_classification/metrics" 
if not os.path.isdir(metrics_directory): 
    os.makedirs(metrics_directory,exist_ok=True)
roc_directory = os.path.join(metrics_directory, "roc")
if not os.path.isdir(roc_directory):
    os.makedirs(roc_directory,exist_ok=True)
scores_directory = os.path.join(metrics_directory, "scores")
if not os.path.isdir(scores_directory):
    os.makedirs(scores_directory,exist_ok=True)

features_filename_list = [CTFM_filename, 
                          FMCIB_filename, 
                          Merlin_filename,
                          ModelsGen_filename,
                          PASTA_filename,
                          SUPREME_filename,
                          VISTA3D_filename,
                          Voco_filename]


##################
### Processing ###
################## 

for features_filename in features_filename_list: 

    with open(features_filename, 'rb') as f: 
        data = pickle.load(f)
    
    # get features and concatenate 
    train_X = [data['train'][i]['feature'] for i in range(len(data['train']))]
    train_X = np.concatenate(train_X,axis=0)
    val_X = [data['val'][i]['feature'] for i in range(len(data['val']))]
    val_X = np.concatenate(val_X,axis=0)
    test_X = [data['test'][i]['feature'] for i in range(len(data['test']))]
    test_X = np.concatenate(test_X,axis=0)
    # get labels
    train_y = [data['train'][i]['row']['label'] for i in range(len(data['train']))]
    val_y = [data['val'][i]['row']['label'] for i in range(len(data['val']))]
    test_y = [data['test'][i]['row']['label'] for i in range(len(data['test']))]

    # ROC curves filename 
    fm_type = os.path.basename(features_filename)
    fm_type = Path(fm_type).stem
    output_png_filename = os.path.join(roc_directory, fm_type + '.png')
    
    # ROC measures filename 
    output_npz_filename = os.path.join(scores_directory, fm_type + ".npz")

    # Training loop with simple hyperparameter search using validation set
    C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    best_val_score = 0
    best_model = None

    for C in C_range:
        linear_model = LogisticRegression(C=C, max_iter=1000)
        linear_model.fit(train_X, train_y)
        val_pred = linear_model.predict_proba(val_X)[:, 1]
        val_score = roc_auc_score(val_y, val_pred)

        print(f"C = {C}: Validation accuracy = {val_score}")

        # Keep track of the best model
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = linear_model

    print(f"Best Validation accuracy: {best_val_score}")

    # Test 
    test_pred = best_model.predict_proba(test_X)[:, 1]
    test_score = roc_auc_score(test_y, test_pred)
    print(f"Score on the testing data: {test_score}")

    # Plot curves 
    plt.figure()
    lw = 2

    split_map = {
        "Train": [train_X, train_y, "steelblue"],
        "Val": [val_X, val_y, "lightblue"],
        "Test": [test_X, test_y, "darkblue"]
    }

    roc_values = [] 
    for split in ["Train", "Val", "Test"]:
        feats, label, color = split_map[split]
        fpr, tpr, thresholds = roc_curve(label, best_model.predict_proba(feats)[:, 1])
        roc_auc = auc(fpr, tpr)
        roc_values.append(roc_auc)
        plt.plot(fpr, tpr, color=color, lw=lw, label=f'{split} ROC curve (area = %0.2f)' % roc_auc, alpha=0.8)

    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--', alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(output_png_filename)
     
    # Save npz file 
    np.savez(output_npz_filename, score=test_score, roc_values=roc_values)