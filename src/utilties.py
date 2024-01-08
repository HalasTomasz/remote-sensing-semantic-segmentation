"""Module with function used to get data and evaluate models"""
import dataset
import test_dataset
import torch.utils.data
import torchvision
import os


def get_loaders(image_dir, mask_dir, batch_size, transformations, num_workers=2, pin_memory=True):

    """Create loaders for models

    Args:
        image_dir (str): image directory
        mask_dir (str): mask directory
        batch_size (int): number of batches
        train_transform (torch.albumentations): transformations of training image
        num_workers (int, optional): number of workers. Defaults to 4.
        pin_memory (bool, optional): pin memory. Defaults to True.

    Returns:
        torch.loader: loader
    """

    dataset_class = dataset.SatelliteImageDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=transformations,
    )

    loader = torch.utils.data.DataLoader(
        dataset_class,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return loader


def get_testing_loader(image_dir, batch_size, transformations, num_workers=2, pin_memory=True):

    """Create loaders for models

    Args:
        image_dir (str): training image directory
        batch_size (int): number of batches
        train_transform (torch.albumentations): transformations of training image
        num_workers (int, optional): number of workers. Defaults to 4.
        pin_memory (bool, optional): pin memory. Defaults to True.

    Returns:
        torch.loader: loader
    """

    dataset_test_class = test_dataset.SatelliteImageDatasetTest(
        image_dir=image_dir,
        transform=transformations,
    )

    loader = torch.utils.data.DataLoader(
        dataset_test_class,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return loader


def check_accuracy_on_cnn_models(loader, model, file_path, device="cuda"):
    """Check accuracy, precision, dice_score, recall and F1 score of our model
        Results will be printed in the end

    Args:
        loader (torch.loader): validation loader
        model (torch.model): CNN model
        device (str, optional): device type. Defaults to "cuda".
    """

    num_correct = 0
    num_pixels = 0
    precision = 0
    dice_score = 0
    recall = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-6)
            true_positives = (preds * y).sum().item()
            false_positives = ((preds == 1) & (y == 0)).sum().item()
            false_negatives = ((preds == 0) & (y == 1)).sum().item()

            precision += true_positives / (true_positives + false_positives + 1e-6)
            recall += true_positives / (true_positives + false_negatives + 1e-6)

    accuracy = num_correct / num_pixels * 100
    dice_score /= len(loader)
    precision /= len(loader)
    recall /= len(loader)

    # F1 Score calculation
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}")
    print(f"Dice score: {dice_score}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

    # Write the results to the file
    with open(file_path, 'a') as file:
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Dice score: {dice_score}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1 Score: {f1_score}\n")
        file.write("\n")

    model.train()


def save_predictions_as_imgs(loader, model, folder, device="cuda"):
    """Make predictions on test images. Crated masks will be saved in given folder

    Args:
        loader (torch.lodader): test loader
        model (torch.model): CNN model
        folder (str): directory where to save created masks
        device (str, optional): device type. Defaults to "cuda".
    """

    if not os.path.exists(folder):
        os.makedirs(folder)

    model.eval()
    for idx, (x) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, os.path.join(folder, f"pred_{idx}.png"))

    model.train()
