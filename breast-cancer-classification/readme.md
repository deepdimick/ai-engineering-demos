# Breast Cancer Classification with PyTorch

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning project implementing binary classification of breast cancer cells (benign vs. malignant) using a fully-connected neural network built with PyTorch.


## Problem Statement

Accurate classification of breast cancer cells is critical for early diagnosis and treatment planning. This project builds a neural network from scratch using PyTorch to classify breast cancer cells based on diagnostic measurements.

## Features

- Binary classification (benign/malignant) implemented in PyTorch
- Data preprocessing pipeline with balanced sampling and standardization
- Custom neural network architecture with configurable layers
- Training/validation workflow with loss tracking
- Comparative experiments with different optimizers (Adam vs. SGD)
- Architecture variations to analyze performance impact

## Quick Start

### Prerequisites

```bash
pip install pandas==2.2.2 numpy==1.26.4 matplotlib==3.8.0 scikit-learn==1.5.0 torch==2.3.1 ucimlrepo==0.0.7
```

### Dataset

The project uses the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) from UCI Machine Learning Repository (CC BY 4.0), automatically fetched via `ucimlrepo`. The dataset contains 569 samples with 30 diagnostic features.

### Run

Open `breast_cancer_classification_pytorch.ipynb` in Jupyter to explore the implementation.

## Model Architecture

| Component | Details |
|-----------|---------|
| **Input Layer** | 30 features (diagnostic measurements) |
| **Hidden Layer** | Dense(64) → ReLU activation |
| **Output Layer** | Dense(2) for binary classification |
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | CrossEntropyLoss |
| **Batch Size** | 2 |
| **Epochs** | 10 |

## Project Structure

```
├── PRACTICE_Neural_Network_for_Breast_Cancer_Classification.ipynb
└── README.md
```

## Results

The model successfully classifies breast cancer cells into benign and malignant categories. Training and test loss curves demonstrate proper convergence over 10 epochs with no signs of overfitting. Experiments with different optimizers (Adam vs. SGD) and varying hidden layer sizes provide insights into architecture design decisions.

## Experimental Variations

The project includes comparative experiments exploring:

1. **Optimizer Comparison**: Implemented both Adam and SGD optimizers with momentum to evaluate convergence characteristics
2. **Architecture Tuning**: Tested varying hidden unit sizes (16, 32, 64) to assess model capacity requirements
3. **Extensibility**: Designed architecture to be easily adapted to other classification datasets

## Skills Demonstrated

- PyTorch neural network architecture design and implementation
- Data preprocessing and class balancing techniques
- Model training, evaluation, and convergence analysis
- Hyperparameter experimentation and comparative analysis
- Scientific visualization of training metrics

## License

MIT

## Acknowledgments

- Project completed as part of IBM AI Engineering Professional Certificate
- Dataset: [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) (CC BY 4.0)
- Framework: PyTorch by Meta AI
