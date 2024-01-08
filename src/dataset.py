"""This module contains class which porpuse is to read satellite images"""

import os
import torch.utils.data
import PIL.Image
import numpy as np


class SatelliteImageDataset(torch.utils.data.Dataset):
    """Class to read image and mask one by one
    and if specfied do transformation on image"""

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # List of file names
        self.image_names = os.listdir(self.image_dir)
        self.mask_names = os.listdir(self.mask_dir)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index):

        image_name = self.image_names[index]
        mask_name = self.mask_names[index]

        # Load images
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(PIL.Image.open(image_path).convert("RGB"))
        mask = np.array(PIL.Image.open(mask_path).convert("L"), dtype=np.float32)

        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
