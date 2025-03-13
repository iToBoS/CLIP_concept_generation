# Zero-Shot Inference on CLIP-Based Models for Dermoscopy

## Overview
This repository contains a concentrated implementation for running zero-shot inference using CLIP-based foundation models on dermoscopy images. It enables retrieving relevant dermatological concepts along with their associated relevance scores.

## Features
- **Zero-Shot Inference**: Utilize CLIP-based models to analyze dermoscopy images without requiring additional training.
- **Concept Retrieval**: Extract relevant dermatological concepts based on image features.
- **Relevance Scoring**: Compute relevance scores for retrieved concepts to assess their significance.

## Requirements
Ensure you have the dependencies installed.
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. Load the model and perform inference:
   ```python
 python inference.py "path/to/dermoscopy_image.jpg"
   ```
## Expected Output
The model returns a dictionary with relevant dermatological concepts and their associated relevance scores:
```json
{
    "linear irregular vessel": {
        "value": "linear irregular vessel",
        "relevance": 0.9053587913513184
    },
    "milky-red areas": {
        "value": "milky-red areas",
        "relevance": 0.7563719749450684
    },
    "arborizing vessel": {
        "value": "arborizing vessel",
        "relevance": 0.7450194358825684
    },
    "white": {
        "value": "white",
        "relevance": 0.7216874957084656
    },
    "red": {
        "value": "red",
        "relevance": 0.7171644568443298
    },
    "text_description": "dermoscopic features detected in the lesion: linear irregular vessel, arborizing vessel, hairpin shape vessel, Parallel Pattern, Structureless Areas, regular streaks, serpentine vessel. colors detected in the lesion: white, red. "

```

## Acknowledgments
This work is based on CLIP-based foundation models (MONET, whylesionclip and open-clip) for medical imaging applications. If you use this repository, consider citing relevant papers on CLIP and dermatology AI research.




