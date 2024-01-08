"""Moduel used for traning purposes"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import ResNet.resnet50_model as resnet_model_module
import utilties


torch.cuda.empty_cache()

LEARNING_RATE = 0.0005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 3
NUM_WORKERS = 4
PIN_MEMORY = True
TRAIN_IMG_DIR = "/home/halas/AerialImageDataset/train/images"
TRAIN_MASK_DIR = "/home/halas/AerialImageDataset/train/gt"
VAL_IMG_DIR = "/home/halas/AerialImageDataset/val/images"
VAL_MASK_DIR = "/home/halas/AerialImageDataset/val/gt"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    """Train CNN model

    Args:
        loader (torch.loader): training loaders
        model (torch.model): chosen CNN model
        optimizer (torch.optimizer): chosen optimizer
        loss_fn (torch.loss_fn): chosen loss function
        scaler (torch.scaler): chosen scaler
    """
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast(enabled=False):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    """Descirbe more
    """
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

    model = resnet_model_module.ResNet50(resnet_model_module.ResNet_Block, [3, 4, 6, 3], 2).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = utilties.get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    print("Done loading train", len(train_loader))
    print("Done loading val", len(val_loader))

    utilties.check_accuracy_on_cnn_models(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"EPOCH NUMER {epoch}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        # check accuracy
        utilties.check_accuracy_on_cnn_models(val_loader, model, device=DEVICE)

    utilties.save_predictions_as_imgs(val_loader, model, "saved_images/unet")


if __name__ == "__main__":
    main()
