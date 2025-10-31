# Feature Extractor Guide

How to create a new dataset feature extractor for TumorImagingBench.

## Overview

The TumorImagingBench framework uses **dataset extractors** to integrate new datasets. Each extractor is a Python module that:

1. **Loads dataset splits** (train/val/test) from CSV files
2. **Validates and preprocesses** each sample
3. **Extracts features** using all available foundation models
4. **Saves results** as pickle files

## Quick Start

To add a new dataset, create a file in `src/tumorimagingbench/evaluation/` following the pattern:

```python
# my_dataset_feature_extractor.py
import pandas as pd
from base_feature_extractor import extract_all_features, save_features
from functools import partial

def get_split_data(split, train_csv, val_csv, test_csv):
    """Load dataset split from CSV."""
    split_paths = {
        "train": train_csv,
        "val": val_csv,
        "test": test_csv
    }
    return pd.read_csv(split_paths[split])

def preprocess_row(row):
    """Validate and preprocess each sample."""
    return row

def extract_features(output_path, train_csv, val_csv, test_csv, model_names=None):
    """Extract features for all models."""
    features = extract_all_features(
        partial(get_split_data, train_csv=train_csv, val_csv=val_csv, test_csv=test_csv),
        preprocess_row,
        model_names=model_names
    )
    save_features(features, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract features for my dataset")
    parser.add_argument("--output", type=str, default="features/my_dataset.pkl")
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--models", type=str, nargs="+", default=None)

    args = parser.parse_args()
    extract_features(args.output, args.train_csv, args.val_csv, args.test_csv, args.models)
```

Usage:
```bash
cd src/tumorimagingbench/evaluation
python my_dataset_feature_extractor.py \
  --output features/my_dataset.pkl \
  --train-csv /path/to/train.csv \
  --val-csv /path/to/val.csv \
  --test-csv /path/to/test.csv
```

---

## Detailed Guide

### Part 1: Dataset Preparation

Before creating a feature extractor, prepare your dataset in CSV format.

#### Expected Directory Structure

```
data/eval/my_dataset/
├── train.csv
├── val.csv
├── test.csv
└── images/
    ├── scan_001.nii.gz
    ├── scan_002.nii.gz
    └── ...
```

#### CSV Format

Each CSV file (train.csv, val.csv, test.csv) should have:

**Required columns:**
- `image_path` (str): Absolute path to NIFTI file (.nii.gz)
- `coordX` (float): X centroid coordinate in physical space (mm)
- `coordY` (float): Y centroid coordinate in physical space (mm)
- `coordZ` (float): Z centroid coordinate in physical space (mm)

**Optional columns:**
- `label` (int/float): Target variable for classification/regression
- Any other metadata (patient_id, scan_date, etc.)

**Example CSV:**

```csv
image_path,coordX,coordY,coordZ,label
/path/to/data/eval/my_dataset/images/scan_001.nii.gz,100.5,150.3,200.1,0
/path/to/data/eval/my_dataset/images/scan_002.nii.gz,110.2,160.8,210.5,1
/path/to/data/eval/my_dataset/images/scan_003.nii.gz,95.8,145.2,195.3,0
```

**Important Notes:**
- All paths must be absolute paths (or relative to the working directory)
- Coordinates should be in physical space (millimeters), not voxel indices
- The coordinates represent the centroid of the region of interest (lesion, tumor, etc.)

---

### Part 2: Create the Feature Extractor

Create a new Python file in `src/tumorimagingbench/evaluation/`:

```bash
touch src/tumorimagingbench/evaluation/my_dataset_feature_extractor.py
```

#### Function 1: `get_split_data(split, ...)`

This function loads and returns the dataset split as a pandas DataFrame.

```python
import pandas as pd

def get_split_data(split, train_csv, val_csv, test_csv):
    """
    Load dataset split from CSV.

    Parameters:
    -----------
    split : str
        One of ['train', 'val', 'test']
    train_csv : str
        Path to training CSV
    val_csv : str
        Path to validation CSV
    test_csv : str
        Path to test CSV

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: image_path, coordX, coordY, coordZ, label (optional), ...

    Raises:
    -------
    ValueError
        If split is not recognized
    FileNotFoundError
        If CSV file not found
    """
    split_paths = {
        "train": train_csv,
        "val": val_csv,
        "test": test_csv
    }

    if split not in split_paths:
        raise ValueError(f"Invalid split: {split}. Must be one of {list(split_paths.keys())}")

    csv_path = split_paths[split]
    return pd.read_csv(csv_path)
```

#### Function 2: `preprocess_row(row)`

This function validates and preprocesses each sample before feature extraction.

**Basic version (no validation):**
```python
def preprocess_row(row):
    """Pass through - no preprocessing needed."""
    return row
```

**Advanced version (with validation):**
```python
import os

def preprocess_row(row):
    """
    Validate and preprocess each sample.

    Parameters:
    -----------
    row : pandas.Series or dict
        Single row from the dataset

    Returns:
    --------
    dict or None
        Preprocessed row (return None to skip the sample)
    """
    # Convert to dict if needed
    if hasattr(row, 'to_dict'):
        row = row.to_dict()

    # 1. Check image file exists
    image_path = row['image_path']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found, skipping: {image_path}")
        return None

    # 2. Check image is NIFTI format
    if not image_path.endswith(('.nii.gz', '.nii')):
        print(f"Warning: Not NIFTI format, skipping: {image_path}")
        return None

    # 3. Validate coordinates are numeric
    try:
        for coord in ['coordX', 'coordY', 'coordZ']:
            float(row[coord])
    except (ValueError, TypeError) as e:
        print(f"Warning: Invalid coordinates in {image_path}, skipping: {e}")
        return None

    # 4. Optional: Validate coordinates are in reasonable range (mm)
    for coord in ['coordX', 'coordY', 'coordZ']:
        if not (-500 <= float(row[coord]) <= 500):
            print(f"Warning: Coordinate {coord}={row[coord]} out of range [-500, 500], skipping")
            return None

    # 5. Handle missing labels
    if 'label' not in row or pd.isna(row['label']):
        row['label'] = -1  # Use -1 for missing labels

    return row
```

#### Function 3: `extract_features(...)`

This is the main entry point that orchestrates feature extraction.

```python
from functools import partial
from base_feature_extractor import extract_all_features, save_features

def extract_features(output_path, train_csv, val_csv, test_csv, model_names=None):
    """
    Extract features for all models.

    Parameters:
    -----------
    output_path : str
        Where to save extracted features (pickle file)
    train_csv : str
        Path to training annotations CSV
    val_csv : str
        Path to validation annotations CSV
    test_csv : str
        Path to test annotations CSV
    model_names : list of str, optional
        Specific models to extract. If None, extracts all available models.
        Example: ['DummyResNetExtractor', 'FMCIBExtractor']

    Returns:
    --------
    None
        Saves results to output_path

    Examples:
    ---------
    >>> # Extract all models
    >>> extract_features('features/my_dataset.pkl',
    ...                   'data/train.csv', 'data/val.csv', 'data/test.csv')
    >>>
    >>> # Extract specific models
    >>> extract_features('features/my_dataset.pkl',
    ...                   'data/train.csv', 'data/val.csv', 'data/test.csv',
    ...                   model_names=['DummyResNetExtractor'])
    """
    print("=" * 70)
    print("TumorImagingBench Feature Extraction - My Dataset")
    print("=" * 70)

    # Create a partial function that binds CSV paths
    get_split_fn = partial(get_split_data, train_csv=train_csv, val_csv=val_csv, test_csv=test_csv)

    # Extract features for all models
    features = extract_all_features(get_split_fn, preprocess_row, model_names=model_names)

    # Save results to disk
    save_features(features, output_path)

    print("=" * 70)
    print("✓ Feature extraction completed successfully")
    print(f"✓ Results saved to {output_path}")
    print("=" * 70)
```

#### Function 4: Command-Line Interface

Add argparse to allow command-line usage:

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract features for My Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Extract using all available models
  python my_dataset_feature_extractor.py \\
    --output features/my_dataset.pkl \\
    --train-csv /path/to/train.csv \\
    --val-csv /path/to/val.csv \\
    --test-csv /path/to/test.csv

  # Extract specific models only
  python my_dataset_feature_extractor.py \\
    --output features/my_dataset.pkl \\
    --train-csv /path/to/train.csv \\
    --val-csv /path/to/val.csv \\
    --test-csv /path/to/test.csv \\
    --models DummyResNetExtractor FMCIBExtractor

  # Check available models
  python -c "from tumorimagingbench.models import get_available_extractors; print(get_available_extractors())"
        """
    )

    parser.add_argument(
        "--output",
        type=str,
        default="features/my_dataset.pkl",
        help="Path where to save extracted features (default: features/my_dataset.pkl)"
    )

    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="Path to training annotations CSV"
    )

    parser.add_argument(
        "--val-csv",
        type=str,
        required=True,
        help="Path to validation annotations CSV"
    )

    parser.add_argument(
        "--test-csv",
        type=str,
        required=True,
        help="Path to test annotations CSV"
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to extract (space-separated). If not specified, uses all available."
    )

    args = parser.parse_args()
    extract_features(args.output, args.train_csv, args.val_csv, args.test_csv, args.models)
```

---

### Part 3: Complete Example

Here's a complete working example for a hypothetical "MyRadiomic" dataset:

```python
"""
MyRadiomic Feature Extractor

Dataset structure:
- data/eval/my_radiomic/
  ├── train.csv
  ├── val.csv
  ├── test.csv
  └── scans/
      ├── patient_001.nii.gz
      └── ...

CSV format:
  image_path,coordX,coordY,coordZ,label
  /path/to/scans/patient_001.nii.gz,100.5,150.3,200.1,0
  ...
"""

import pandas as pd
import os
from pathlib import Path
from base_feature_extractor import extract_all_features, save_features
from functools import partial


def get_split_data(split, train_csv, val_csv, test_csv):
    """Load dataset split from CSV."""
    split_paths = {
        "train": train_csv,
        "val": val_csv,
        "test": test_csv
    }

    if split not in split_paths:
        raise ValueError(f"Invalid split: {split}")

    csv_path = split_paths[split]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Verify required columns
    required_cols = ['image_path', 'coordX', 'coordY', 'coordZ']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")

    return df


def preprocess_row(row):
    """Validate and preprocess each sample."""
    # Convert to dict if needed
    if hasattr(row, 'to_dict'):
        row = row.to_dict()

    # Verify image exists
    image_path = row['image_path']
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return None

    # Verify NIFTI format
    if not image_path.endswith(('.nii.gz', '.nii')):
        print(f"Warning: Not NIFTI format: {image_path}")
        return None

    # Verify coordinates
    try:
        for coord in ['coordX', 'coordY', 'coordZ']:
            float(row[coord])
    except (ValueError, TypeError):
        print(f"Warning: Invalid coordinates: {image_path}")
        return None

    return row


def extract_features(output_path, train_csv, val_csv, test_csv, model_names=None):
    """Extract features for all models."""
    print("=" * 70)
    print("MyRadiomic Feature Extraction")
    print("=" * 70)

    # Extract using all models
    features = extract_all_features(
        partial(get_split_data, train_csv=train_csv, val_csv=val_csv, test_csv=test_csv),
        preprocess_row,
        model_names=model_names
    )

    # Save results
    save_features(features, output_path)

    print("=" * 70)
    print(f"✓ Features extracted for {len(features)} models")
    print(f"✓ Saved to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract features for MyRadiomic dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python my_radiomic_feature_extractor.py \\
    --output features/my_radiomic.pkl \\
    --train-csv data/eval/my_radiomic/train.csv \\
    --val-csv data/eval/my_radiomic/val.csv \\
    --test-csv data/eval/my_radiomic/test.csv

  python my_radiomic_feature_extractor.py \\
    --output features/my_radiomic.pkl \\
    --train-csv data/eval/my_radiomic/train.csv \\
    --val-csv data/eval/my_radiomic/val.csv \\
    --test-csv data/eval/my_radiomic/test.csv \\
    --models DummyResNetExtractor
        """
    )

    parser.add_argument("--output", type=str, default="features/my_radiomic.pkl")
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--models", type=str, nargs="+", default=None)

    args = parser.parse_args()
    extract_features(args.output, args.train_csv, args.val_csv, args.test_csv, args.models)
```

---

## Usage Examples

### Example 1: Command Line with Default Paths

```bash
cd src/tumorimagingbench/evaluation

python my_dataset_feature_extractor.py \
  --output features/my_dataset.pkl \
  --train-csv /home/suraj/Repositories/TumorImagingBench/data/eval/my_dataset/train.csv \
  --val-csv /home/suraj/Repositories/TumorImagingBench/data/eval/my_dataset/val.csv \
  --test-csv /home/suraj/Repositories/TumorImagingBench/data/eval/my_dataset/test.csv
```

### Example 2: Extract Specific Models Only

```bash
python my_dataset_feature_extractor.py \
  --output features/my_dataset.pkl \
  --train-csv data/train.csv \
  --val-csv data/val.csv \
  --test-csv data/test.csv \
  --models DummyResNetExtractor FMCIBExtractor
```

### Example 3: Programmatic Usage

```python
from src.tumorimagingbench.evaluation.my_dataset_feature_extractor import extract_features

extract_features(
    output_path='features/my_dataset.pkl',
    train_csv='data/train.csv',
    val_csv='data/val.csv',
    test_csv='data/test.csv',
    model_names=['DummyResNetExtractor']
)
```

---

## Reference Implementation

For reference, see these existing extractors:

- **Simple example**: `dummy_dataset_feature_extractor.py` - Basic template with minimal preprocessing
- **Real dataset**: `luna_feature_extractor.py` - LUNA16 lung nodule dataset
- **Complex example**: `nsclc_radiomics_feature_extractor.py` - NSCLC radiomics dataset with labels

---

## Troubleshooting

### Issue: "Image not found" errors during preprocessing

**Solution**: Verify that paths in your CSV are absolute paths or correctly relative to where you run the script.

```bash
# Check a few paths
head data/train.csv
ls /path/to/first/image/in/csv
```

### Issue: "Invalid coordinates" errors

**Solution**: Ensure coordinates in CSV are numeric and in physical space (mm), not voxel indices.

```python
# Example: convert voxel indices to physical coordinates
# (assuming 1mm isotropic spacing, which is default)
coordX = voxel_x * spacing_x
```

### Issue: "Module not found" when importing

**Solution**: Make sure you're running from the evaluation directory or adjusting Python path:

```bash
cd src/tumorimagingbench/evaluation
python my_dataset_feature_extractor.py ...
```

Or:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/src/tumorimagingbench/evaluation"
python my_dataset_feature_extractor.py ...
```

### Issue: GPU out of memory

**Solution**: Reduce batch size or extract models sequentially. Check `base_feature_extractor.py` for batch size configuration.

### Issue: Some models fail to load

**Solution**: Check which models are available and filter to available ones:

```python
from tumorimagingbench.models import get_available_extractors
print(get_available_extractors())

# Then use --models flag to specify only available models
python my_dataset_feature_extractor.py ... --models DummyResNetExtractor
```

---

## Best Practices

1. **Always validate input**: Check that images exist and coordinates are valid
2. **Return None to skip**: If a sample is invalid, return None from `preprocess_row()` to skip it gracefully
3. **Provide useful logging**: Print warnings/errors so users know what's happening
4. **Use absolute paths**: Store absolute paths in CSVs or document relative path expectations
5. **Document your format**: Add a docstring explaining your CSV format and any special requirements
6. **Test locally first**: Run with `--models DummyResNetExtractor` on a small subset to test
7. **Keep CSVs simple**: Use the minimal required columns; add extra metadata as needed

---

## Files Generated

After running feature extraction, you'll get a pickle file with structure:

```python
{
    'ModelName1': {
        'train_features': numpy array (N_train, feature_dim),
        'val_features': numpy array (N_val, feature_dim),
        'test_features': numpy array (N_test, feature_dim),
    },
    'ModelName2': {
        'train_features': ...,
        'val_features': ...,
        'test_features': ...,
    },
    ...
}
```

Load and use:

```python
import pickle

with open('features/my_dataset.pkl', 'rb') as f:
    features = pickle.load(f)

# Access features for a specific model
model_name = 'DummyResNetExtractor'
train_features = features[model_name]['train_features']  # Shape: (N_samples, feature_dim)
```

---

## See Also

- **Getting Started**: [00_getting_started.ipynb](./00_getting_started.ipynb)
- **Model Integration**: [01_model_integration.ipynb](./01_model_integration.ipynb)
- **API Reference**: [03_api_reference.ipynb](./03_api_reference.ipynb)
- **Base Extractor**: `src/tumorimagingbench/evaluation/base_feature_extractor.py`
