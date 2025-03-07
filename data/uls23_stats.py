# Create CSV file to store label metrics
import csv
import numpy as np
from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk

data_dir = Path("/mnt/data1/datasets/ULS23")
csv_file = data_dir / "label_metrics.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "label_name",
            "size_cm3",
            "max_length_x_mm",
            "max_length_y_mm",
            "max_length_z_mm",
        ]
    )

    label_files = list((data_dir / "annotations").rglob("*.nii.gz"))
    for label_fn in tqdm(label_files, desc="Processing labels"):
        label = sitk.ReadImage(str(label_fn))
        # Get size of mask in cm^3
        label_arr = sitk.GetArrayFromImage(label)
        mask_voxels = (label_arr > 0).sum()  # Count non-zero voxels
        voxel_volume = np.prod(label.GetSpacing())  # Volume of one voxel in mm^3
        label_size_cm3 = (mask_voxels * voxel_volume) / 1000  # Convert to cm^3

        # Get max length in each dimension in mm
        spacing = label.GetSpacing()
        nonzero_points = np.nonzero(label_arr)
        max_lengths = []
        for dim, sp in zip(nonzero_points, spacing):
            length_mm = (dim.max() - dim.min() + 1) * sp
            max_lengths.append(length_mm)

        writer.writerow(
            [
                label_fn,
                f"{label_size_cm3:.2f}",
                f"{max_lengths[2]:.2f}",  # x dimension
                f"{max_lengths[1]:.2f}",  # y dimension
                f"{max_lengths[0]:.2f}",  # z dimension
            ]
        )
