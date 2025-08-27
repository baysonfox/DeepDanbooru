from typing import Any, Union
import os

import six
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import deepdanbooru as dd

from .dataset import load_image_records, load_tags
from .dataset_wrapper import DatasetWrapper


def load_image_for_evaluate(
    input_: Union[str, six.BytesIO], width: int, height: int, normalize: bool = True
) -> Any:
    if isinstance(input_, six.BytesIO):
        image = Image.open(input_)
    else:
        image = Image.open(input_)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize with aspect ratio preservation (similar to TF's AREA method)
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image = np.array(image)
    
    # Transform and pad image to exact dimensions
    image = dd.image.transform_and_pad_image(image, width, height)

    if normalize:
        image = image / 255.0

    return image
