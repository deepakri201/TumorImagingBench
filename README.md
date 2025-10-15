# TumorImagingBench

<div align="center">

**A comprehensive framework for evaluating and comparing foundation model feature extractors for radiomics in medical imaging.**

</div>

## 📋 Overview

TumorImagingBench is a robust platform that enables researchers and practitioners to:

* Extract meaningful features from medical images using state-of-the-art foundation models
* Compare performance metrics across diverse radiomics datasets
* Systematically evaluate model stability, robustness, and interpretability 
* Benchmark novel foundation models against established approaches

This framework bridges the gap between advancing foundation models and their practical application in medical imaging analysis.

## 🔍 Key Features

* **Unified Interface**: Common API for all foundation model extractors
* **Comprehensive Evaluation**: Standardized metrics across multiple datasets
* **Interpretability Tools**: Generation of saliency maps and attribution analysis
* **Extensible Architecture**: Easily integrate new models and datasets

## 📂 Repository Structure

```
FM-extractors-radiomics/
├── models/              # Foundation model implementations
├── notebooks/           
│   ├── modelling/       # Dataset-specific modeling notebooks
│   └── analysis/        # Performance, robustness, and stability analysis
├── scripts/             # Utility scripts for batch processing
├── data/                # Dataset directory (not tracked in git)
├── utils/               # Utility functions for data processing
└── evaluation/          # Evaluation metrics and protocols
```

## 🧠 Supported Foundation Models

| Model | Description |
|-------|-------------|
| **FMCIB** | Foundation Model for Cancer Image Biomarkers |
| **CT-FM** | CT Foundation Model |
| **CT-CLIP-ViT** | CT-specific CLIP Vision Transformer |
| **PASTA** | Pathology and Radiology Image Analysis Model |
| **VISTA3D** | 3D Vision Transformer for Medical Imaging |
| **Voco** | Volumetric Contrastive Learning Model |
| **SUPREM** | Supervised Pretraining for Medical Imaging |
| **Merlin** | Multi-modal Embedding for Radiology and Learning |
| **MedImageInsight** | Medical Image Understanding Framework |
| **ModelsGen** | Generative Foundation Models for Medical Imaging |

## 📊 Supported Datasets

* **LUNA16**: Lung Nodule Analysis
* **DLCS**: Duke Lung Cancer Dataset
* **NSCLC Radiomics**: Non-Small Cell Lung Cancer
* **NSCLC Radiogenomics**: Radiogenomic Analysis of NSCLC
* **C4KC-KiTs**: Clear Cell Renal Cell Carcinoma Kidney Tumor Segmentation
* **Colorectal Liver Metastases**: Liver Metastases Dataset

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/AIM-Harvard/TumorImagingBench.git
cd TumorImagingBench

# Install dependencies
uv sync
```

## 🚀 Usage

### Feature Extraction

```python
from models import CTClipVitExtractor, FMCIBExtractor

# Initialize a model
model = FMCIBExtractor()
model.load()

# Extract features from a sample
features = model.extract(sample_path)
```

For systematic feature extraction across datasets, we provide dedicated scripts in the `evaluation/` directory. These scripts offer a standardized approach that can be extended to new datasets through our base feature extractor class.

### Model Evaluation

For examples of model evaluation on different datasets, explore the notebooks in the `notebooks/modelling/` directory. These notebooks demonstrate:

* Feature extraction workflows
* Model training and validation
* Performance analysis and comparison
* Visualization of results

## 📈 Analysis Tools

Our repository includes specialized analysis notebooks:

| Notebook | Purpose |
|----------|---------|
| `stability_analysis.ipynb` | Evaluate model stability with various perturbations |
| `robustness_analysis.ipynb` | Assess model robustness to noise and transformations |
| `saliency_analysis.ipynb` | Visualize and analyze model activation maps |
| `overall_analysis.ipynb` | Compare aggregate performance across models and datasets |

## 🤝 Contributing

We welcome contributions to improve this framework! Here's how you can contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

* Follow the existing code style and documentation patterns
* Add tests for new functionality
* Update documentation to reflect changes
* Ensure backward compatibility where possible

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@article{TumorImagingBench,
  title={Foundation model embeddings for quantitative tumor imaging biomarkers},
  author={}, 
  journal={},
  year={},
  volume={},
  pages={},
  publisher={}
}
```

## 📄 License

This project is licensed under the [LICENSE NAME] - see the LICENSE file for details.
