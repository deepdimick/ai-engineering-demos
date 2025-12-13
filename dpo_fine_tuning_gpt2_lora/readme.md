# DPO Fine-Tuning with LoRA

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch&logoColor=fff)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A parameter-efficient fine-tuning pipeline that aligns GPT-2 with human preferences using Direct Preference Optimization (DPO) and Low-Rank Adaptation (LoRA).

## Problem Statement

Aligning language models with human preferences traditionally requires complex reinforcement learning pipelines with separate reward models. DPO simplifies this by directly optimizing on preference data, while LoRA enables efficient fine-tuning on consumer hardware.

## Features

- Direct Preference Optimization without reward model training
- Parameter-efficient fine-tuning using LoRA adapters
- Automatic GPU/CPU detection with optional 4-bit quantization
- Training visualization (loss curves)
- Side-by-side comparison of base vs. fine-tuned outputs

## Quick Start

### Prerequisites

```bash
pip install torch==2.9.1 transformers>=4.57.0 trl>=0.25.0 peft>=0.14.0 accelerate datasets matplotlib pandas
```

For GPU quantization (optional):
```bash
pip install bitsandbytes
```

### Dataset

The notebook uses [UltraFeedback Binarized](https://huggingface.co/datasets/BarraHome/ultrafeedback_binarized) from Hugging Face, which downloads automatically.

### Run

Open `DPO_Fine_Tuning_GPT2_LoRA.ipynb` in Jupyter and run all cells.

## Model Architecture

| Component | Details |
|-----------|---------|
| **Base Model** | GPT-2 (124M parameters) |
| **Adapter** | LoRA (r=4, α=8, dropout=0.1) |
| **Target Modules** | c_attn, c_proj |
| **Optimizer** | AdamW (lr=1e-4) |
| **Loss** | DPO Loss (β=0.2) |
| **Quantization** | 4-bit NF4 (GPU only, optional) |

## Project Structure

```
├── DPO_Fine_Tuning_GPT2_LoRA.ipynb
├── README.md
└── dpo/
    └── checkpoint-*/
```

## Results

The fine-tuned model demonstrates improved alignment with human preferences compared to the base GPT-2 model. See the notebook for training curves and sample comparisons.

## Skills Demonstrated

- Direct Preference Optimization (DPO)
- Parameter-efficient fine-tuning (PEFT/LoRA)
- Hugging Face Transformers & TRL ecosystem
- Mixed-precision training & quantization

## License

MIT

## Acknowledgments

- Project completed as part of IBM AI Engineering Professional Certificate
- Dataset: [UltraFeedback](https://huggingface.co/datasets/BarraHome/ultrafeedback_binarized) (Hugging Face)
- Libraries: Hugging Face Transformers, TRL, PEFT
