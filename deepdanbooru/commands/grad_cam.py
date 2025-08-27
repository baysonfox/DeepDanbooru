import os

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import deepdanbooru as dd
from scipy import ndimage


def get_gradient(model, x, output_mask):
    """Get gradients for Grad-CAM using PyTorch"""
    x.requires_grad_(True)
    
    output = model(x)
    gradcam_loss = torch.sum(output_mask * output)
    
    # Compute gradients
    gradcam_loss.backward()
    gradients = x.grad
    
    return gradients


def norm_clip_grads(grads):
    upper_quantile = np.quantile(grads, 0.99)
    lower_quantile = np.quantile(grads, 0.01)
    clipped_grads = np.abs(np.clip(grads, lower_quantile, upper_quantile))

    return clipped_grads / np.max(clipped_grads)


def filter_grads(grads):
    return ndimage.median_filter(grads, 10)


def to_onehot(length, index):
    """Create one-hot tensor for PyTorch"""
    value = torch.zeros(1, length, dtype=torch.float32)
    value[0, index] = 1.0
    return value


def grad_cam(project_path, target_path, output_path, threshold):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(target_path):
        raise Exception(f"Target path {target_path} is not exists.")

    if os.path.isfile(target_path):
        taget_image_paths = [target_path]
    else:
        patterns = [
            "*.[Pp][Nn][Gg]",
            "*.[Jj][Pp][Gg]",
            "*.[Jj][Pp][Ee][Gg]",
            "*.[Gg][Ii][Ff]",
        ]

        taget_image_paths = dd.io.get_file_paths_in_directory(target_path, patterns)

        taget_image_paths = dd.extra.natural_sorted(taget_image_paths)

    model = dd.project.load_model_from_project(project_path)
    model = model.to(device)
    model.eval()
    
    tags = dd.project.load_tags_from_project(project_path)
    # Use default dimensions since PyTorch models don't have input_shape attribute
    width = 299  # Default width, should be configurable
    height = 299  # Default height, should be configurable

    dd.io.try_create_directory(output_path)

    for image_path in taget_image_paths:
        image = dd.data.load_image_for_evaluate(image_path, width=width, height=height)
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        image_folder = os.path.join(output_path, image_name)
        dd.io.try_create_directory(image_folder)

        Image.fromarray(np.uint8(image * 255.0)).save(
            os.path.join(image_folder, f"input.png")
        )
        image_for_result = image
        
        # Convert to PyTorch tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Run inference
        with torch.no_grad():
            y = model(image_tensor)[0].cpu().numpy()

        result_dict = {}

        estimated_tags = []

        for i, tag in enumerate(tags):
            result_dict[tag] = y[i]

            if y[i] >= threshold:
                estimated_tags.append((i, tag))

        print(f"Tags of {image_path}:")

        for tag in tags:
            if result_dict[tag] >= threshold:
                print(f"({result_dict[tag]:05.3f}) {tag}")

        image = image.astype(np.float32)

        for estimated_tag in estimated_tags:
            print(f"Calculating grad-cam ... ({estimated_tag[1]})")
            
            # Create input tensor for gradient computation
            image_grad_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
            image_grad_tensor.requires_grad_(True)
            
            # Create one-hot target
            target_onehot = to_onehot(len(tags), estimated_tag[0]).to(device)
            
            # Get gradients
            grads = get_gradient(model, image_grad_tensor, target_onehot)
            grads = grads[0].cpu().detach().numpy().transpose(1, 2, 0)  # CHW -> HWC
            
            print("Normalizing gradients ...")
            grads = norm_clip_grads(grads)
            print("Filtering gradients ...")
            grads = filter_grads(grads)
            Image.fromarray(np.uint8(grads * 255.0)).save(
                os.path.join(
                    image_folder,
                    f"result-{estimated_tag[1]}.png".replace(":", "_").replace(
                        "/", "_"
                    ),
                )
            )
            mask_array = np.stack([np.max(grads, axis=-1)] * 3, axis=2)
            Image.fromarray(
                np.uint8(np.multiply(image_for_result, mask_array) * 255.0)
            ).save(
                os.path.join(
                    image_folder,
                    f"result-{estimated_tag[1]}-masked.png".replace(":", "_").replace(
                        "/", "_"
                    ),
                )
            )
