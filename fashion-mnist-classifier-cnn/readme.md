# Fashion MNIST Classification

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive PyTorch project that classifies Fashion MNIST images using convolutional neural networks with options for both standard and batch-normalized architectures.

## Problem Statement

Fashion MNIST classification is a fundamental computer vision task that extends beyond the original handwritten digit MNIST dataset. This project demonstrates core deep learning concepts including CNN architecture design, data pipeline construction, model training, and performance evaluation on a 10-class image classification problem.

## Features

- Two CNN architectures: standard CNN and batch-normalized CNN variant
- Complete data pipeline with image preprocessing and DataLoader integration
- Training and validation workflow with performance metrics
- Visualization of cost and accuracy curves across epochs
- Configurable model parameters (output channels, kernel sizes, learning rates)

## Quick Start

### Prerequisites

```bash
pip install torch torchvision matplotlib pillow
```

### Dataset

The Fashion MNIST dataset is automatically downloaded on first run via PyTorch's `torchvision.datasets` module. No manual setup required.

### Run

Open `fashion_mnist_classification.ipynb` in Jupyter and execute all cells sequentially.

## Model Architecture

Two CNN implementations are provided:

| Component                 | Standard CNN                          | Batch-Normalized CNN                              |
| ------------------------- | ------------------------------------- | ------------------------------------------------- |
| **Conv Layer 1**    | Conv2d(1→16, 5×5) + ReLU + MaxPool  | Conv2d(1→16, 5×5) + BatchNorm + ReLU + MaxPool  |
| **Conv Layer 2**    | Conv2d(16→32, 5×5) + ReLU + MaxPool | Conv2d(16→32, 5×5) + BatchNorm + ReLU + MaxPool |
| **Fully Connected** | Linear(32×4×4→10)                  | Linear(32×4×4→10) + BatchNorm                  |
| **Optimizer**       | SGD (lr=0.1)                          | SGD (lr=0.1)                                      |
| **Loss Function**   | Cross Entropy Loss                    | Cross Entropy Loss                                |

## Project Structure

```
├── FashionMNISTProject-Complete_2.ipynb
├── README.md
└── fashion/data/
    ├── train/
    └── test/
```

## Training Configuration

- **Batch Size:** 100
- **Epochs:** 5
- **Image Size:** 16×16 pixels
- **Classes:** 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot)
- **Dataset Split:** 60,000 training samples, 10,000 validation samples

## Results

The model produces training cost and validation accuracy curves across epochs. Batch normalization variant typically achieves faster convergence and better stability compared to the standard CNN.

## Skills Demonstrated

- PyTorch fundamentals (tensors, modules, optimizers)
- Convolutional neural network design and implementation
- Custom dataset handling with transforms and DataLoaders
- Model training loops with optimization and backpropagation
- Performance visualization and analysis
- Batch normalization techniques

## License

MIT

## Acknowledgments

- Project completed as part of IBM AI Engineering Professional Certificate
- Dataset:[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) by Zalando Research (MIT License)
- Based on IBM Deep Learning course materials
