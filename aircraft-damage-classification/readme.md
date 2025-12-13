# Aircraft Damage Classification & Captioning

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=fff)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning pipeline that classifies aircraft surface damage (dent vs. crack) using transfer learning with VGG16, and generates natural language descriptions using the BLIP transformer model.

## Problem Statement

Manual aircraft inspection is time-consuming and prone to human error. This project automates damage detection using computer vision and adds interpretability through natural language captioning.

## Features

- Binary image classification (dent/crack) using fine-tuned VGG16
- Automated image captioning and summarization with BLIP
- Custom Keras layer for transformer integration
- Training visualization (accuracy/loss curves)

## Quick Start

### Prerequisites

```bash
pip install tensorflow==2.17.1 torch transformers pillow matplotlib scikit-learn
```

### Dataset

Download the [Aircraft Damage Dataset](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk) (CC BY 4.0) and extract to `aircraft_damage_dataset_v1/`.

### Run

Open `vgg16_aircraft_damage_detection.ipynb` in Jupyter and run all cells.

## Model Architecture

| Component | Details |
|-----------|---------|
| **Base Model** | VGG16 (ImageNet weights, frozen) |
| **Classifier** | Dense(512) → Dropout(0.3) → Dense(512) → Dropout(0.3) → Sigmoid |
| **Captioning** | BLIP (Salesforce/blip-image-captioning-base) |
| **Optimizer** | Adam (lr=0.0001) |
| **Loss** | Binary Crossentropy |

## Project Structure

```
├── Final_Project_Classification_and_Captioning.ipynb
├── README.md
└── aircraft_damage_dataset_v1/
    ├── train/
    ├── valid/
    └── test/
```

## Results

The model classifies aircraft damage images and generates descriptive captions. See the notebook for training curves and sample predictions.

## Skills Demonstrated

- Transfer learning & feature extraction
- Custom Keras layer implementation
- Multi-modal AI (vision + language)
- Data pipeline design

## License

MIT

## Acknowledgments

- Project completed as part of IBM AI Engineering Professional Certificate
- Dataset: [Roboflow Aircraft Dataset](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk) by Youssef Donia (CC BY 4.0)
- Models: VGG16, BLIP (Salesforce)
