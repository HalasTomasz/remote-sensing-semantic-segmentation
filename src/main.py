"""Moduel used for traning purposes"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import utilties
import UNet.unet_model
import ResNet.resnet50_model
import train_CNN
import train_forest

torch.cuda.empty_cache()

LEARNING_RATE = 0.005
BATCH_SIZE = 1
NUM_WORKERS = 2
PIN_MEMORY = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_TRAIN_IMG_DIR = "/home/halas/AerialImageDatasetPATCH/train/images"
PATCH_TRAIN_MASK_DIR = "/home/halas/AerialImageDatasetPATCH/train/gt"
PATCH_VAL_IMG_DIR = "/home/halas/AerialImageDatasetPATCH/val/images"
PATCH_VAL_MASK_DIR = "/home/halas/AerialImageDatasetPATCH/val/gt"
PATCH_TEST_IMG_DIR = "/home/halas/AerialImageDatasetPATCH/test/images"

MODEL_TYPE = "FOREST"


def main():
    """Main function responsible for choosing models"""

    train_transform = A.Compose(
        [
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_loader = utilties.get_loaders(
        PATCH_TRAIN_IMG_DIR,
        PATCH_TRAIN_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    val_loader = utilties.get_loaders(
        PATCH_VAL_IMG_DIR,
        PATCH_VAL_MASK_DIR,
        BATCH_SIZE,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    test_loader = utilties.get_testing_loader(
        PATCH_TEST_IMG_DIR,
        BATCH_SIZE,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    print("Done loading train:", len(train_loader))
    print("Done loading val:", len(val_loader))
    print("Done loading test:", len(test_loader))

    if MODEL_TYPE == "UNET":
        model = UNet.unet_model.UNETModel(in_channels=3, out_channels=1).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_CNN.traning_cnn(model, optimizer, loss_fn, train_loader, val_loader, test_loader, "UNET")

    elif MODEL_TYPE == "RESNET":
        model = ResNet.resnet50_model.ResNet50(ResNet.resnet50_model.ResNet50, [3, 4, 6, 3], 1).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_CNN.traning_cnn(model, optimizer, loss_fn, train_loader, val_loader, test_loader, "RESNET50")

    else:

        model = train_forest.train_random_forest(train_loader)
        train_forest.validate_model(val_loader, model)
        train_forest.test_model(test_loader, model)


if __name__ == "__main__":
    main()
