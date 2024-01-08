"""This module contains class which porpuse is to read satellite images"""

import os
import torch.utils.data
import PIL.Image
import numpy as np


class SatelliteImageDatasetTest(torch.utils.data.Dataset):
    """Class to read image and mask one by one
    and if specfied do transformation on image"""

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # List of file names
        self.image_names = os.listdir(self.image_dir)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index):

        image_name = self.image_names[index]

        # Load images
        image_path = os.path.join(self.image_dir, image_name)

        image = np.array(PIL.Image.open(image_path).convert("RGB"))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image
