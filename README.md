# FM-extractors-radiomics

A framework for evaluating and comparing different foundation model extractors for radiomics in medical imaging.

## Overview

This repository provides tools, models, and evaluation scripts to:

1. Extract features from medical images using various foundation models
2. Compare performance across different radiomics datasets
3. Analyze the stability, robustness, and interpretability of these models
4. Facilitate benchmarking of new foundation models against established ones

## Repository Structure

- `models/`: Implementations of various foundation model extractors
  - CT-CLIP, FMCIB, PASTA, VISTA3D, MedImageInsight, and more
- `notebooks/`:
  - `modelling/`: Notebooks for modeling on different medical datasets
    - LUNA, DLCS, NSCLC Radiomics, NSCLC Radiogenomics, C4KC-KiTs, Colorectal Liver Metastases
  - `analysis/`: Analysis of model performance, robustness, stability, and saliency
- `scripts/`: Utility scripts for batch processing and analysis
  - `generate_saliency_maps.py`: Generate saliency maps for model interpretability
- `data/`: Directory for datasets (not tracked in git)
- `utils/`: Utility functions for data processing and analysis
- `evaluation/`: Evaluation metrics and protocols

## Supported Models

- FMCIB (Foundation Model for Cancer Image Biomarkers)
- CT-FM (CT Foundation Model)
- CT-CLIP-ViT
- PASTA
- VISTA3D
- Voco
- SUPREM
- Merlin
- MedImageInsight
- ModelsGen

## Supported Datasets

- LUNA16
- DLCS (Duke Lung Cancer Dataset)
- NSCLC Radiomics
- NSCLC Radiogenomics
- C4KC-KiTs (Clear Cell Renal Cell Carcinoma Kidney Tumor Segmentation)
- Colorectal Liver Metastases

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FM-extractors-radiomics.git
cd FM-extractors-radiomics

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Feature Extraction

To extract features from medical images using the implemented models:

```python
from models import CTClipVitExtractor, FMCIBExtractor

# Initialize a model
model = FMCIBExtractor()
model.load()

# Extract features from a sample
features = model.extract(sample_path)
```
Feature extraction is offered through systematic scripts for each dataset provided in [evaluation](./evaluation/). A base feature extractor can be extended for a new dataset. 

### Model Evaluation

Explore the notebooks in the `notebooks/modelling/` directory to see examples of model evaluation on different datasets.

## Analysis

The repository includes several analysis notebooks:

- `stability_analysis.ipynb`: Evaluate model stability to different perturbations
- `robustness_analysis.ipynb`: Assess model robustness
- `saliency_analysis.ipynb`: Visualize and analyze model saliency maps
- `overall_analysis.ipynb`: Aggregate performance analysis across datasets and models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this framework in your research, please cite:

```
[Citation information to be added]
```
