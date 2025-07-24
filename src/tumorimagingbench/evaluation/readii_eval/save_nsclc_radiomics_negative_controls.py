from readii.negative_controls import applyNegativeControl
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from fmcib.run import get_features


df = pd.read_csv("/home/suraj/Repositories/FM-extractors-radiomics/data/eval/nsclc_radiomics/annotations.csv")

row_list = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_path = row['image_path']
    mask_path = image_path.replace("CT.nrrd", "masks/GTV-1.nrrd")
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    shuffled_non_roi = applyNegativeControl(image, "shuffled", "non_roi", mask, randomSeed=42)
    sitk.WriteImage(shuffled_non_roi, image_path.replace("CT.nrrd", "CT_shuffled-non_roi.nrrd"))

    shuffled_roi = applyNegativeControl(image, "shuffled", "roi", mask, randomSeed=42) 
    sitk.WriteImage(shuffled_roi, image_path.replace("CT.nrrd", "CT_shuffled-roi.nrrd"))

    randomized_sampled_non_roi = applyNegativeControl(image, "randomized_sampled", "non_roi", mask, randomSeed=42)
    sitk.WriteImage(randomized_sampled_non_roi, image_path.replace("CT.nrrd", "CT_randomized_sampled-non_roi.nrrd"))

    randomized_sampled_roi = applyNegativeControl(image, "randomized_sampled", "roi", mask, randomSeed=42)
    sitk.WriteImage(randomized_sampled_roi, image_path.replace("CT.nrrd", "CT_randomized_sampled-roi.nrrd"))


    row["shuffled-non_roi"] = image_path.replace("CT.nrrd", "CT_shuffled-non_roi.nrrd")
    row["shuffled-roi"] = image_path.replace("CT.nrrd", "CT_shuffled-roi.nrrd")
    row["randomized_sampled-non_roi"] = image_path.replace("CT.nrrd", "CT_randomized_sampled-non_roi.nrrd")
    row["randomized_sampled-roi"] = image_path.replace("CT.nrrd", "CT_randomized_sampled-roi.nrrd")

    row_list.append(row)

updated_df = pd.DataFrame(row_list)

shuffled_non_roi_df = updated_df.rename(
    columns={"image_path": "original", "shuffled-non_roi": "image_path"}
)
shuffled_roi_df = updated_df.rename(
    columns={"image_path": "original", "shuffled-roi": "image_path"}
)
randomized_sampled_non_roi_df = updated_df.rename(
    columns={"image_path": "original", "randomized_sampled-non_roi": "image_path"}
)
randomized_sampled_roi_df = updated_df.rename(
    columns={"image_path": "original", "randomized_sampled-roi": "image_path"}
)
shuffled_non_roi_df.to_csv("negative_control_shuffled_non_roi.csv")
shuffled_roi_df.to_csv("negative_control_shuffled_roi.csv")
randomized_sampled_non_roi_df.to_csv("negative_control_randomized_sampled_non_roi.csv")
randomized_sampled_roi_df.to_csv("negative_control_randomized_sampled_roi.csv")
print("Merging features for shuffled non-ROI...")
shuffled_non_roi_df = pd.merge(
    shuffled_non_roi_df,
    get_features("negative_control_shuffled_non_roi.csv"),
    on="image_path"
)

print("Merging features for shuffled ROI...")
shuffled_roi_df = pd.merge(
    shuffled_roi_df,
    get_features("negative_control_shuffled_roi.csv"),
    on="image_path"
)

print("Merging features for randomized sampled non-ROI...")
randomized_sampled_non_roi_df = pd.merge(
    randomized_sampled_non_roi_df,
    get_features("negative_control_randomized_sampled_non_roi.csv"),
    on="image_path"
)

print("Merging features for randomized sampled ROI...")
randomized_sampled_roi_df = pd.merge(
    randomized_sampled_roi_df,
    get_features("negative_control_randomized_sampled_roi.csv"),
    on="image_path"
)

print("Saving merged features for shuffled non-ROI...")
shuffled_non_roi_df.to_csv("negative_control_shuffled_non_roi.csv")
print("Saving merged features for shuffled ROI...")
shuffled_roi_df.to_csv("negative_control_shuffled_roi.csv")
print("Saving merged features for randomized sampled non-ROI...")
randomized_sampled_non_roi_df.to_csv("negative_control_randomized_sampled_non_roi.csv")
print("Saving merged features for randomized sampled ROI...")
randomized_sampled_roi_df.to_csv("negative_control_randomized_sampled_roi.csv")
