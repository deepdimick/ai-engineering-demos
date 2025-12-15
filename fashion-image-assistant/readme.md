# Fashion Image Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=fff)
![IBM watsonx](https://img.shields.io/badge/IBM_watsonx-052FAD?logo=ibm&logoColor=fff)
![License](https://img.shields.io/badge/License-MIT-green)

A multi-modal AI pipeline that encodes fashion images using ResNet50 for similarity matching, then generates natural language style analysis using Llama 3.2 Vision model.


## Problem Statement

Finding similar fashion items and understanding outfit composition requires both visual similarity matching and contextual understanding. This project combines computer vision embeddings with vision-language models to identify matching catalog items and generate professional style descriptions.

## Features

- Image encoding and feature extraction using ResNet50 V2 weights
- Cosine similarity matching against a fashion catalog dataset
- Natural language outfit analysis via Llama 3.2 Vision Instruct
- Support for both URL and local file image inputs
- Configurable similarity thresholds for match quality

## Quick Start

### Prerequisites

```bash
pip install torch torchvision pillow requests numpy scikit-learn ibm-watsonx-ai
```

### Configuration

Set your IBM watsonx.ai credentials:

```python
api_key = "your-api-key"
project_id = "your-project-id"
```

### Run

```python
from image_processor import ImageProcessor
from llm_service import LlamaVisionService

# Initialize components
processor = ImageProcessor()
llm = LlamaVisionService(
    model_id="meta-llama/llama-3-2-90b-vision-instruct",
    project_id=project_id,
    api_key=api_key
)

# Process image and find matches
result = processor.encode_image("path/to/image.jpg", is_url=False)
matched_row, score = processor.find_closest_match(result['vector'], catalog_df)

# Generate style analysis
response = llm.generate_fashion_response(
    result['base64'], catalog_df, score
)
```

## Model Architecture

| Component | Details |
|-----------|---------|
| **Feature Extractor** | ResNet50 (ImageNet V2 weights) |
| **Preprocessing** | ResNet50_Weights.IMAGENET1K_V2 transforms |
| **Similarity** | Cosine similarity on flattened feature vectors |
| **Vision LLM** | Llama 3.2 90B Vision Instruct |
| **LLM Parameters** | temperature=0.2, top_p=0.6, max_tokens=2000 |

## Project Structure

```
├── image_processor.py    # Image encoding and similarity matching
├── llm_service.py        # IBM watsonx.ai Llama Vision integration
└── README.md
```

## Results

The pipeline identifies visually similar fashion items from a catalog and generates professional retail-style descriptions including item details, style categorization, and color/pattern analysis.

## Skills Demonstrated

- Transfer learning with pretrained CNN models
- Feature extraction and vector similarity search
- Multi-modal AI (vision + language)
- IBM watsonx.ai API integration
- Production-ready class design patterns

## License

MIT

## Acknowledgments

- Project completed as part of IBM AI Engineering Professional Certificate
- Models: ResNet50 (PyTorch), Llama 3.2 Vision (Meta)
