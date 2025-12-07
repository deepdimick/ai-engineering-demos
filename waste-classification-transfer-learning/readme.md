# Waste Classification Using Transfer Learning

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=fff)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning pipeline that automates waste classification using transfer learning with VGG16 to distinguish between recyclable and organic waste materials.

<!-- Optional: Add a screenshot or demo gif here -->
<!-- ![Demo](images/demo.png) -->

## Problem Statement

Manual waste sorting is labor-intensive and prone to errors, leading to contamination of recyclable materials. This project automates the waste classification process using computer vision and transfer learning to improve efficiency and reduce contamination rates in waste management systems.

## Features

- Binary image classification (recyclable/organic) using VGG16 transfer learning
- Two model approaches: feature extraction and fine-tuning
- Data augmentation for improved model robustness
- Comprehensive training visualization (accuracy/loss curves)
- Model evaluation with classification reports

## Quick Start

### Prerequisites

```bash
pip install tensorflow==2.17.0 numpy==1.26.0 scikit-learn==1.5.1 matplotlib==3.9.2
```

### Dataset

The project uses the [Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data) from Kaggle. The dataset is automatically downloaded and extracted when running the notebook.

### Run

Open `Final_Proj-Classify_Waste_Products_Using_TL_FT.ipynb` in Jupyter and run all cells.

## Model Architecture

| Component | Details |
|-----------|---------|
| **Base Model** | VGG16 (ImageNet weights, frozen layers) |
| **Feature Extraction** | Dense(512) → Dropout(0.3) → Dense(512) → Dropout(0.3) → Sigmoid |
| **Fine-Tuning** | Unfroze block5_conv3 + custom dense layers |
| **Optimizer** | RMSprop (lr=0.0001) with exponential decay |
| **Loss** | Binary Crossentropy |
| **Input Size** | 150x150x3 |

## Project Structure

```
├── Final_Proj-Classify_Waste_Products_Using_TL_FT.ipynb
├── README.md
└── o-vs-r-split/
    ├── train/
    │   ├── O/  (Organic)
    │   └── R/  (Recyclable)
    ├── valid/
    │   ├── O/
    │   └── R/
    └── test/
        ├── O/
        └── R/
```

## Results

The project implements two transfer learning approaches:

1. **Feature Extraction Model**: Uses frozen VGG16 layers as feature extractor
2. **Fine-Tuned Model**: Unfreezes final convolutional block for improved performance

Both models are evaluated on test data with classification reports and visual predictions. See the notebook for detailed training curves and sample predictions.

## Skills Demonstrated

- Transfer learning & feature extraction with VGG16
- Model fine-tuning techniques
- Image data augmentation and preprocessing
- Early stopping and model checkpointing
- Learning rate scheduling
- Performance visualization and evaluation

## License

MIT

## Acknowledgments

- Project completed as part of IBM AI Engineering Professional Certificate
- Dataset: [Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data) by techsash
- Pre-trained model: VGG16 (ImageNet weights)
