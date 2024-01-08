import os
import numpy as np
import patchify
import PIL.Image

TEST_IMG_DIR = "/home/halas/AerialImageDataset/test/images"
PATCH_TEST_IMG_DIR = "/home/halas/AerialImageDatasetPATCH/test/images"


def patchify_images(image_dir, destination_image_dir):
    """Main object of this function is to patchify input images

    Args:
        image_dir (str):  image directory
        destination_image_dir (str): dir to save patchfied images
    """
    # If it doesn't exist, create it
    if not os.path.exists(destination_image_dir):
        os.makedirs(destination_image_dir)
        print(f"Directory {destination_image_dir} created.")

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)

        image = np.array(PIL.Image.open(image_path).convert("RGB"))

        img_patches = patchify.patchify(image, (512, 512, 3), step=512)

        img_patches = img_patches.reshape(-1, 512, 512, 3)

        # Create a unique name for each patch based on the original image name and patch index
        for i, (img_patch) in enumerate(img_patches):
            patch_name = f"{image_name.split('.')[0]}_patch_{i}.tif"

            # Save image patch
            img_patch_path = os.path.join(destination_image_dir, patch_name)
            PIL.Image.fromarray(img_patch.astype(np.uint8)).save(img_patch_path)


patchify_images(TEST_IMG_DIR, PATCH_TEST_IMG_DIR)
print("Image patches and masks saved")
