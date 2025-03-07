import pandas as pd
import numpy as np
from readii.process.label import setPatientIdAsIndex
from readii.io.writers.correlation_writer import CorrelationWriter

from readii.analyze.correlation import getFeatureCorrelations, getSelfCorrelations

import itertools
import matplotlib.pyplot as plt
from pathlib import Path

from analyze import prepPatientIndex, makeAllClusterHeatmapPlots, makeAllHeatmapPlots, makeAllHistogramPlots

from sklearn.cluster import AgglomerativeClustering 

negative_control_dict = {
    "negative_control_randomized_sampled_non_roi": pd.read_csv("../evaluation/readii_eval/negative_control_randomized_sampled_non_roi.csv"),
    "negative_control_randomized_sampled_roi": pd.read_csv("../evaluation/readii_eval/negative_control_randomized_sampled_roi.csv"),
    "negative_control_shuffled_non_roi": pd.read_csv("../evaluation/readii_eval/negative_control_shuffled_non_roi.csv"),
    "negative_control_shuffled_roi": pd.read_csv("../evaluation/readii_eval/negative_control_shuffled_roi.csv"),
}
vertical_features_df = pd.read_csv("/home/suraj/Repositories/FM-extractors-radiomics/evaluation/readii_eval/original.csv")

vertical_features_df = prepPatientIndex(vertical_features_df, "image_path", "LUNG1-[0-9]{3}").filter(like="pred")

for key, feature_df in negative_control_dict.items():
    negative_control_dict[key] = prepPatientIndex(feature_df, "image_path", "LUNG1-[0-9]{3}").filter(like="pred")

correlation_method = "pearson"
heatmap_cmap = "nipy_spectral"
overwrite = False

# initialize clustering id array
clustering = np.array([])

# Iterate over each negative control feature set and perform correlation analysis
for key, horizontal_features_df in negative_control_dict.items():
    print(f"Processing {key} correlations.")
    # Calculate correlations between original image features and image type features
    feature_correlation_df = getFeatureCorrelations(vertical_features=vertical_features_df,
                                                    horizontal_features=horizontal_features_df,
                                                    vertical_feature_name="Original",
                                                    horizontal_feature_name=key,
                                                    method = correlation_method)
    print("Generating heatmaps for correlations.")
    vert_heatmap_path, horiz_heatmap_path, cross_heatmap_path = makeAllHeatmapPlots(feature_correlation_df,
                                                                                    "Original",
                                                                                    key,
                                                                                    "./readii_results/heatmap",
                                                                                    correlation_method,
                                                                                    heatmap_cmap,
                                                                                    overwrite)
    if len(clustering) == 0:
        # Cluster the features based on the correlations from the Original image
        original_corr = getSelfCorrelations(feature_correlation_df, "Original")
        clustering = AgglomerativeClustering(linkage="complete", metric="precomputed", n_clusters = None, distance_threshold = 0).fit_predict(original_corr)
    
    vert_heatmap_path, horiz_heatmap_path, cross_heatmap_path = makeAllClusterHeatmapPlots(feature_correlation_df,
                                                                                        "Original",
                                                                                        key,
                                                                                        clustering,
                                                                                        "./readii_results/clustered",
                                                                                        correlation_method,
                                                                                        heatmap_cmap,
                                                                                        overwrite)
    
    print("Generating histograms for correlations.")
    vert_histogram_path, horiz_histogram_path, cross_histogram_path = makeAllHistogramPlots(feature_correlation_df,
                                                                                            "Original",
                                                                                            key,
                                                                                            "./readii_results/hist",
                                                                                            correlation_method,
                                                                                            num_bins=450,
                                                                                            self_corr_y_max = 250000,
                                                                                            cross_corr_y_max = 950000,
                                                                                            overwrite=overwrite)    



