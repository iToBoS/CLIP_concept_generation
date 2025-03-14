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
## Docker Support

This project includes a Dockerfile, allowing you to run the application inside a Docker container for a consistent runtime environment. To build and run the Docker container, use the following commands:
```bash
# Build the Docker image
docker build -t conceptm .

# Run the container
docker run --rm -v $(pwd):/app conceptm
```
## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd CLIP_concept_generation
   ```
2. Enter the path to your data folder and perform inference:
```bash
 python inference.py path/to/dermoscopy_image.jpg
 ```
If you leave the data path empy it will automatically read the images in the data folder.

You can update the inference code in line 22 and 23 and use the different model architcture and threshold for the relevance score:
```python
model_api = "whylesion" #monet , clip
threshold = 0.6
```
If you want to use your own concept(s) you can simply create a json file similar to what we have in ```iToBoS_concepts.json``` and update the code in line 385:
```python
concepts_dictionary = json.load(open("iToBoS_concepts.json", "r"))

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
This work is based on CLIP-based foundation models ([MONET](https://github.com/suinleelab/MONET), [whylesionclip](https://github.com/YueYANG1996/KnoBo)  and [open-clip](https://github.com/openai/CLIP)) for medical imaging applications. If you use this repository, consider citing relevant papers on CLIP and dermatology AI research.




