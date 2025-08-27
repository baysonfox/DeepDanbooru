import os
from typing import Any, Iterable, List, Tuple, Union

import six
import torch
import numpy as np

import deepdanbooru as dd

def save_txt_file(txt_path, list):
    last_index = len(list)-1
    last_tag = list[last_index]
    with open(txt_path, 'w') as writer:
        for i in list:
            if last_tag == i:
                writer.write(i)
                writer.close()
            else:
                writer.write(i + ", ")
    print("Saved text file.")

def evaluate_image(
    image_input: Union[str, six.BytesIO], model: Any, tags: List[str], threshold: float
) -> Iterable[Tuple[str, float]]:
    # Get model input dimensions (assuming standard input shape)
    # For PyTorch models, we need to check the model's expected input size
    # This is a simplified approach - in a real implementation, you'd store this info
    width = 299  # Default width, should be configurable
    height = 299  # Default height, should be configurable

    image = dd.data.load_image_for_evaluate(image_input, width=width, height=height)

    # Convert to PyTorch tensor and add batch dimension
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    
    # Set model to evaluation mode
    model.eval()
    
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        y = model(image_tensor)[0].cpu().numpy()

    result_dict = {}

    for i, tag in enumerate(tags):
        result_dict[tag] = y[i]

    for tag in tags:
        if result_dict[tag] >= threshold:
            yield tag, result_dict[tag]


def evaluate(
    target_paths, #this
    project_path,
    model_path,
    tags_path,
    threshold,
    allow_gpu,
    compile_model,
    allow_folder,
    save_txt,
    folder_filters,
    verbose,
):
    # Handle GPU settings for PyTorch
    if not allow_gpu:
        device = torch.device('cpu')
        print("Using CPU for inference")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

    if not model_path and not project_path:
        raise Exception("You must provide project path or model path.")

    if not tags_path and not project_path:
        raise Exception("You must provide project path or tags path.")

    target_image_paths = []

    for target_path in target_paths:
        if allow_folder and not os.path.isfile(target_path):
            target_image_paths.extend(
                dd.io.get_image_file_paths_recursive(target_path, folder_filters)
            )
        else:
            target_image_paths.append(target_path)

    target_image_paths = dd.extra.natural_sorted(target_image_paths)

    if model_path:
        if verbose:
            print(f"Loading model from {model_path} ...")
        # Load PyTorch model
        model = torch.load(model_path, map_location=device)
        model = model.to(device)
    else:
        if verbose:
            print(f"Loading model from project {project_path} ...")
        model = dd.project.load_model_from_project(
            project_path, compile_model=compile_model
        )
        model = model.to(device)

    if tags_path:
        if verbose:
            print(f"Loading tags from {tags_path} ...")
        tags = dd.data.load_tags(tags_path)
    else:
        if verbose:
            print(f"Loading tags from project {project_path} ...")
        tags = dd.project.load_tags_from_project(project_path)

    for image_path in target_image_paths:
        print(f"Tags of {image_path}:") #yup!
        if save_txt: tag_list = []
        for tag, score in evaluate_image(image_path, model, tags, threshold):
            print(f"({score:05.3f}) {tag}")
            if save_txt: tag_list.append(tag)
        if save_txt:
            txt_file_path = str(os.path.splitext(image_path)[0]) + ".txt"
            save_txt_file(txt_file_path, tag_list)
        print()
