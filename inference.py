import torch
import torchvision.transforms as T
import clip
import os
import pandas as pd
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from pathlib import Path
import glob
from PIL import Image
import scipy
import numpy as np
import json
import time
import sys
import open_clip
import argparse
 #sudo docker run -v $(pwd)/output:/app/output conceptm
 #python inference.py data
start = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cuda" if torch.cuda.is_available() else "cpu"
model_api = "whylesion"
threshold = 0.60
#################################################################################################
if len(sys.argv)>1:
        image_dir = sys.argv[1]
#################################################################################################
class ImageDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for loading images and associated metadata.

    Args:
        image_path_list (list): A list of file paths to the images.
        transform (callable): A function/transform to apply to the images.
        metadata_df (pandas.DataFrame, optional): A pandas DataFrame containing metadata for the images.

    Raises:
        AssertionError: If the length of `image_path_list` is not equal to the length of `metadata_df`.

    Returns:
        dict: A dictionary containing the image and metadata (if available) for a given index.

    """

    def __init__(self, image_path_list, transform, metadata_df=None):
        self.image_path_list = image_path_list
        self.transform = transform
        self.metadata_df = metadata_df

        if self.metadata_df is None:
            self.metadata_df = pd.Series(index=self.image_path_list)
        else:
            assert len(self.image_path_list) == len(
                self.metadata_df
            ), "image_path_list and metadata_df must have the same length"
            self.metadata_df.index = self.image_path_list

    def __getitem__(self, idx):
        image = Image.open(self.image_path_list[idx])

        ret = {"image": self.transform(image)}

        if self.metadata_df is not None:
            ret.update({"metadata": self.metadata_df.iloc[idx]})

        return ret

    def __len__(self):
        return len(self.image_path_list)
##################################################################################################  
def custom_collate(batch):
    """Custom collate function for the dataloader.

    Args:
        batch (list): list of dictionaries, each dictionary is a batch of data

    Returns:
        dict: dictionary of collated data
    """

    ret = {}
    for key in batch[0]:
        if isinstance(batch[0][key], pd.Series):
            try:
                ret[key] = pd.concat([d[key] for d in batch], axis=1).T
            except RuntimeError:
                raise RuntimeError(f"Error while concatenating {key}")
        else:
            try:
                ret[key] = torch.utils.data.dataloader.default_collate(
                    [d[key] for d in batch]
                )
            except RuntimeError:
                raise RuntimeError(f"Error while concatenating {key}")

    return ret
def custom_collate_per_key(batch_all):
    """Custom collate function batched outputs.

    Args:
        batch_all (dict): dictionary of lists of objects, each dictionary is a batch of data
    Returns:
        dict: dictionary of collated data
    """

    ret = {}
    for key in batch_all:
        if isinstance(batch_all[key][0], pd.DataFrame):
            ret[key] = pd.concat(batch_all[key], axis=0)
        elif isinstance(batch_all[key][0], torch.Tensor):
            ret[key] = torch.concat(batch_all[key], axis=0)
        else:
            print(f"Collating {key}...")
            ret[key] = torch.utils.data.dataloader.default_collate(
                [elem for batch in batch_all[key] for elem in batch]
            )

    return ret

def dataloader_apply_func(
dataloader, func, collate_fn=custom_collate_per_key, verbose=True
):
    """Apply a function to a dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): torch dataloader
        func (function): function to apply to each batch
        collate_fn (function, optional): collate function. Defaults to custom_collate_batch.

    Returns:
        dict: dictionary of outputs
    """
    func_out_dict = {}

    for batch in dataloader:
        for key, func_out in func(batch).items():
            func_out_dict.setdefault(key, []).append(func_out)

    return collate_fn(func_out_dict)

def get_transform(n_px):
    def convert_image_to_rgb(image):
        return image.convert("RGB")
    return T.Compose(
        [
            T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(n_px),
            convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),        
        ]
    )

if model_api=="clip":
    # Load model using original clip implementation
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)[0], get_transform(n_px=224)
    map_location = torch.device(device)
    model.load_state_dict(torch.hub.load_state_dict_from_url("https://aimslab.cs.washington.edu/MONET/weight_clip.pt", map_location=map_location))
    model.eval()
    print("model was loaded using original clip implementation")
elif model_api=="whylesion":
    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:yyupenn/whylesionclip")
    model.to(device)
    model.eval()
else:
    # Load model using huggingface clip implementation
    preprocess = get_transform(n_px=224)
    processor_hf = AutoProcessor.from_pretrained("chanwkim/monet")
    model_hf = AutoModelForZeroShotImageClassification.from_pretrained("chanwkim/monet")
    model_hf.to(device)
    model_hf.eval()
    model = model_hf
    print("MONET model was loaded")  

#print(os.path.exists(image_dir))
image_path_list = [Path(path) for path in glob.glob(str(Path(image_dir) / "*"))]
image_path_list = [
    image_path
    for image_path in image_path_list
    if image_path.suffix in [".png", ".jpg", ".jpeg"]
]
image_name_list = [image_path.name for image_path in image_path_list]
image_dataset = ImageDataset(
    image_path_list, preprocess, 
)
print(f"Number of images: {len(image_dataset)}")
dataloader = torch.utils.data.DataLoader(
    image_dataset,
    batch_size=64,
    num_workers=1,
    collate_fn=custom_collate,
    shuffle=False,
)

def batch_func(batch):
    with torch.no_grad():
        if model_api=="clip" or model_api=="whylesion":
            image_features = model.encode_image(batch["image"].to(device))
        else:
            image_features = model_hf.get_image_features(batch["image"].to(device))

    return {
        "image_features": image_features.detach().cpu(),
        "metadata": batch["metadata"],
    }

image_embedding = dataloader_apply_func(
    dataloader=dataloader,
    func=batch_func,
    collate_fn=custom_collate_per_key,
)
def get_prompt_embedding(
    concept_term_list=[],
    prompt_template_list=[
        "This is skin image of {}",
        "This is dermatology image of {}",
        "This is image of {}",
    ],
    prompt_ref_list=[
        ["This is skin image"],
        ["This is dermatology image"],
        ["This is image"],
    ],
):
    """
    Generate prompt embeddings for a concept

    Args:
        concept_term_list (list): List of concept terms that will be used to generate prompt target embeddings.
        prompt_template_list (list): List of prompt templates.
        prompt_ref_list (list): List of reference phrases.

    Returns:
        dict: A dictionary containing the normalized prompt target embeddings and prompt reference embeddings.
    """
    # target embedding
    prompt_target = [
        [prompt_template.format(term) for term in concept_term_list]
        for prompt_template in prompt_template_list
    ]

    if model_api=="clip" or model_api=="monet":
        prompt_target_tokenized = [
            clip.tokenize(prompt_list, truncate=True) for prompt_list in prompt_target
        ]
    else: #whylesion
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        prompt_target_tokenized = [ 
            tokenizer(prompt_list) for prompt_list in prompt_target
        ]
    with torch.no_grad():
        if model_api=="clip" or model_api=="whylesion":
            prompt_target_embedding = torch.stack(
            [
                model.encode_text(prompt_tokenized.to(next(model.parameters()).device)).detach().cpu()
                for prompt_tokenized in prompt_target_tokenized
            ])
        else:
            prompt_target_embedding = torch.stack(
                [
                    model_hf.get_text_features(prompt_tokenized.to(next(model.parameters()).device)).detach().cpu()
                    for prompt_tokenized in prompt_target_tokenized
                ]
            )
    prompt_target_embedding_norm = (
        prompt_target_embedding / prompt_target_embedding.norm(dim=2, keepdim=True)
    )

    # reference embedding
    if model_api=="clip" or model_api=="monet":
        prompt_ref_tokenized = [
            clip.tokenize(prompt_list, truncate=True) for prompt_list in prompt_ref_list
        ]
    else: #whylesion
        prompt_ref_tokenized = [ 
            tokenizer(prompt_list) for prompt_list in prompt_ref_list]
        
    with torch.no_grad():
        if model_api=="clip" or model_api=="whylesion":
            prompt_ref_embedding = torch.stack(
            [
                model.encode_text(prompt_tokenized.to(next(model.parameters()).device)).detach().cpu()
                for prompt_tokenized in prompt_ref_tokenized
            ])
        else:
            prompt_ref_embedding = torch.stack(
                [               
                    model_hf.get_text_features(prompt_tokenized.to(next(model.parameters()).device)).detach().cpu()
                    for prompt_tokenized in prompt_ref_tokenized
                ]
            )
    prompt_ref_embedding_norm = prompt_ref_embedding / prompt_ref_embedding.norm(
        dim=2, keepdim=True
    )

    return {
        "prompt_target_embedding_norm": prompt_target_embedding_norm,
        "prompt_ref_embedding_norm": prompt_ref_embedding_norm,
    }

def calculate_concept_presence_score(
    image_features_norm,
    prompt_target_embedding_norm,
    prompt_ref_embedding_norm,
    temp=1 / np.exp(4.5944),
):
    """
    Calculates the concept presence score based on the given image features and concept embeddings.

    Args:
        image_features_norm (numpy.Tensor): Normalized image features.
        prompt_target_embedding_norm (torch.Tensor): Normalized concept target embedding.
        prompt_ref_embedding_norm (torch.Tensor): Normalized concept reference embedding.
        temp (float, optional): Temperature parameter for softmax. Defaults to 1 / np.exp(4.5944).

    Returns:
        np.array: Concept presence score.
    """

    target_similarity = (
        prompt_target_embedding_norm.float() @ image_features_norm.T.float()
    )
    ref_similarity = prompt_ref_embedding_norm.float() @ image_features_norm.T.float()

    target_similarity_mean = target_similarity.mean(dim=[1])
    ref_similarity_mean = ref_similarity.mean(axis=1)

    concept_presence_score = scipy.special.softmax(
        [target_similarity_mean.numpy() / temp, ref_similarity_mean.numpy() / temp],
        axis=0,
    )[0, :].mean(axis=0)

    return concept_presence_score

dermo_features =  [
    "irregular streaks",
    "regular streaks",
    "typical pigment Network",
    "atypical pigment Network",
    "regression areas",
    "blue areas regression",
    "combinations of regression",
    "white regression areas",
    "scar-like depigmentation",
    "peppering or granularity regression",
    "regular globules",
    "irregular globules",
    "regular dots",
    "irregular dots",
    "blue-white veil",
    "diffuse regular pigmentation",
    "diffuse irregular pigmentation",
    "localized regular Pigmentation",
    "localized irregular pigmentation",
    "milky-red areas",
    "polymorphous vessel",
    "corkscrew vessel",
    "wreath vessel",
    "within regression vessel",
    "linear irregular vessel",
    "comma shape Vessel",
    "hairpin shape vessel",
    "arborizing vessel",
    "dotted vessel",
    "serpentine vessel",
    "coiled or glomerular vessel",
    "leaf-like Structure",
    "large blue-gray ovoid nests",
    "Inflammation",
    "Milia-like Cyst",
    "Multiple Blue-gray Globule",
    "Parallel Pattern",
    "Ulceration",
    "Structureless Areas",
    "Blotches"

]
elevation_features = ["flat","raised"]
other_features = [ "Artifact", "Hair", "Dermoscopy Ring"]
color_features = [ "red","white","black","light-brown","dark-brown","blue-gray"]
shape_features = ["symmetric in both axes","symmetrical in one axis","asymmetrical"]
border_features = ["regular border", "irregular border"]

image_features_norm_batch = image_embedding["image_features"] / image_embedding["image_features"].norm(dim=1, keepdim=True)
concepts_dictionary = json.load(open("iToBoS_concepts.json", "r"))
results = {}
for i, image_name in enumerate(image_name_list):
    #print(f"{image_name}")  # Include image name in print
    image_features_norm = image_features_norm_batch[i]  # Extract features for the current image
    #################################################################################################
    results = {}
    for concept, terms in concepts_dictionary.items():
        concept_embedding = get_prompt_embedding(concept_term_list=terms)
        concept_presence_score = calculate_concept_presence_score(
            prompt_target_embedding_norm=concept_embedding["prompt_target_embedding_norm"],
            image_features_norm=image_features_norm,  # Use features for the current image
            prompt_ref_embedding_norm=concept_embedding["prompt_ref_embedding_norm"],
        )

        results[concept] = {"value": concept.lower(), "relevance": float(concept_presence_score)}
    sorted_concepts = dict(sorted(results.items(), key=lambda  item: item[1]['relevance'], reverse=True))
    #print(sorted_results)
    dermo_detected = []
    color_detected = []
    shape_detected = []
    border_detected = []
    elevation_detected = []
    other_detected = []
  
    for item_key, item_data in sorted_concepts.items():
        relevance = item_data.get("relevance") 
        if relevance is not None and relevance >= threshold:  
            if item_key in dermo_features:
                dermo_detected.append(item_key)

            if item_key in color_features:
                color_detected.append(item_key)

            if item_key in shape_features:
                shape_detected.append(item_key)

            if item_key in border_features:
                border_detected.append(item_key)

            if item_key in elevation_features:
                elevation_detected.append(item_key)

            if item_key in other_features:
                other_detected.append(item_key)

    text = "" ""    
    if len(dermo_detected)>0:
        text += f"dermoscopic features detected in the lesion: {', '.join(dermo_detected)}. "
    if len(color_detected)>0:
        text += f"colors detected in the lesion: {', '.join(color_detected)}. "
    if len(shape_detected)>0:
        text += f"symmetry features: {', '.join(shape_detected)}. "
    if len(border_detected)>0:
        text += f"the lesion has {border_detected[0]}. "
    if len(elevation_detected)>0:
        text += f"the lesion appears to be {elevation_detected[0]}. "
    if len(other_detected)>0:
        text += f"other information detected in the lesion: {', '.join(other_detected)}. "
    if text == "":
        text = "No textual features detected for this lesion."
    
    sorted_concepts["text_description"] =  text
       
    #print(text)   
    json_filename = f"output/{image_name.split('.')[0]}_concept_scores.json"
    with open(json_filename, "w") as json_file:
        json.dump(sorted_concepts, json_file, indent=4)
        print(f"Results saved to {json_filename}")

print(f"Time taken: {time.time()-start} seconds.")