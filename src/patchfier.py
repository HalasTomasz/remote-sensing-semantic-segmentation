"""Patchify images to smaller patches to used them"""
import os
import numpy as np
import patchify
import PIL.Image

TRAIN_IMG_DIR = "/home/halas/AerialImageDataset/train/images"
TRAIN_MASK_DIR = "/home/halas/AerialImageDataset/train/gt"
VAL_IMG_DIR = "/home/halas/AerialImageDataset/val/images"
VAL_MASK_DIR = "/home/halas/AerialImageDataset/val/gt"

PATCH_TRAIN_IMG_DIR = "/home/halas/AerialImageDatasetPATCH/train/images"
PATCH_TRAIN_MASK_DIR = "/home/halas/AerialImageDatasetPATCH/train/gt"
PATCH_VAL_IMG_DIR = "/home/halas/AerialImageDatasetPATCH/val/images"
PATCH_VAL_MASK_DIR = "/home/halas/AerialImageDatasetPATCH/val/gt"


def patchify_images(image_dir, mask_dir, destination_image_dir, destination_mask_dir):
    """Main object of this function is to patchify input images

    Args:
        image_dir (str):  image directory
        mask_dir (str):  mask directory
        destination_image_dir (str): dir to save patchfied images
        destination_mask_dir (str): dir to save patchfied masks
    """
    #  If it doesn't exist, create it
    if not os.path.exists(destination_image_dir) or not os.path.exists(destination_mask_dir):
        os.makedirs(destination_image_dir)
        print(f"Directory {destination_image_dir} created.")
        os.makedirs(destination_mask_dir)
        print(f"Directory {destination_mask_dir} created.")

    for (image_name, mask_name) in zip(os.listdir(image_dir), os.listdir(mask_dir)):
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, mask_name)

        image = np.array(PIL.Image.open(image_path).convert("RGB"))
        mask = np.array(PIL.Image.open(mask_path).convert("L"))

        img_patches = patchify.patchify(image, (512, 512, 3), step=512).reshape(-1, 512, 512, 3)
        mask_patches = patchify.patchify(mask, (512, 512), step=512).reshape(-1, 512, 512)

        #  Create a unique name for each patch based on the original image name and patch index
        for i, (img_patch, mask_patch) in enumerate(zip(img_patches, mask_patches)):
            patch_name = f"{image_name.split('.')[0]}_patch_{i}.tif"

            #  Save image patch
            img_patch_path = os.path.join(destination_image_dir, patch_name)
            PIL.Image.fromarray(img_patch.astype(np.uint8)).save(img_patch_path)

            #  Save mask patch
            mask_patch_path = os.path.join(destination_mask_dir, patch_name)
            PIL.Image.fromarray(mask_patch.astype(np.uint8)).save(mask_patch_path)


patchify_images(TRAIN_IMG_DIR, TRAIN_MASK_DIR, PATCH_TRAIN_IMG_DIR, PATCH_TRAIN_MASK_DIR)
patchify_images(VAL_IMG_DIR, VAL_MASK_DIR, PATCH_VAL_IMG_DIR, PATCH_VAL_MASK_DIR)

print("Image patches and masks saved")
