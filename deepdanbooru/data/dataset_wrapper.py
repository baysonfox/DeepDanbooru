import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

import deepdanbooru as dd


class DatasetWrapper(Dataset):
    """
    PyTorch Dataset wrapper for data pipelining/augmentation.
    """

    def __init__(
        self, inputs, tags, width, height, scale_range, rotation_range, shift_range
    ):
        self.inputs = inputs
        self.width = width
        self.height = height
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.tag_all_array = np.array(tags)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image_path, tag_string = self.inputs[idx]
        
        # Load and process image
        image = self.load_image(image_path)
        
        # Get tags for this image
        labels = self.load_tags_for_image(tag_string)
        
        return image, labels

    def get_dataloader(self, minibatch_size, shuffle=True, num_workers=4):
        return DataLoader(
            self, 
            batch_size=minibatch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

    def load_image(self, image_path):
        """Load and preprocess image from path"""
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            # Return a default image if loading fails
            image = Image.new('RGB', (self.width, self.height), color='black')
        
        # Convert to numpy for processing
        image = np.array(image)
        
        # Apply transformations
        if self.scale_range:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
        else:
            scale = None

        if self.rotation_range:
            rotation = random.uniform(self.rotation_range[0], self.rotation_range[1])
        else:
            rotation = None

        if self.shift_range:
            shift_x = random.uniform(self.shift_range[0], self.shift_range[1])
            shift_y = random.uniform(self.shift_range[0], self.shift_range[1])
            shift = (shift_x, shift_y)
        else:
            shift = None

        # Apply image transformations (rotation, scale, shift, padding)
        image = dd.image.transform_and_pad_image(
            image=image,
            target_width=self.width,
            target_height=self.height,
            rotation=rotation,
            scale=scale,
            shift=shift,
        )

        # Normalize to 0-1 range
        image = image / 255.0
        
        # Convert to PyTorch tensor (HWC -> CHW)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image

    def load_tags_for_image(self, tag_string):
        """Convert tag string to multi-label array"""
        if isinstance(tag_string, bytes):
            tag_string = tag_string.decode()
        
        tag_array = np.array(tag_string.split(" "))
        
        labels = np.where(np.isin(self.tag_all_array, tag_array), 1, 0).astype(
            np.float32
        )
        
        return torch.from_numpy(labels)