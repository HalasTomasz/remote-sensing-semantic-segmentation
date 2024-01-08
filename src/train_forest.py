"""Module to train, validate and test Random Forest Algortihm"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import random_forest_features


def train_random_forest(train_loader):
    """Fit traing data to Random Forest Classifier

    Args:
        train_loader (torch.loader): loader with training data

    Returns:
        (sklearn.model): Random Forest Classifier model
    """
    list_of_df = []
    mask_list = []

    for _, (data, targets) in enumerate(train_loader):

        data = data.squeeze().permute(1, 2, 0).numpy()

        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        tmp_df = random_forest_features.features.calc_features(data)
        list_of_df.append(tmp_df)
        mask_list.append(targets.numpy().reshape(-1))

    combined_df = pd.concat(list_of_df, ignore_index=True)
    combined_mask = pd.Series(np.concatenate(mask_list))
    list_of_df = []
    mask_list = []  # save space
    model = RandomForestClassifier(n_estimators=60, random_state=42)
    model.fit(combined_df, combined_mask)

    return model


def validate_model(val_loader, model):
    """Validate model and print results

    Args:
        val_loader (torch.loader): loader with validation data
        (sklearn.model): Random Forest Classifier model
    """
    list_of_df = []
    mask_list = []

    for _, (data, targets) in enumerate(val_loader):

        data = data.squeeze().permute(1, 2, 0).numpy()

        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        tmp_df = random_forest_features.features.calc_features(data)
        list_of_df.append(tmp_df)
        mask_list.append(targets.numpy().reshape(-1))

    combined_df = pd.concat(list_of_df, ignore_index=True)
    combined_mask = pd.Series(np.concatenate(mask_list))

    predictions = model.predict(combined_df)
    print(confusion_matrix(combined_mask, predictions))
    report = classification_report(combined_mask, predictions)
    print(report)


def test_model(test_loader, model):
    """Test model and save results

    Args:
        test_loader (torch.loader): loader with validation data
        (sklearn.model): Random Forest Classifier model
    """

    for _, (data, image_name) in enumerate(test_loader):
        data = data.squeeze().permute(1, 2, 0).numpy()
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        tmp_df = random_forest_features.features.calc_features(data)
        predictions = model.predict(tmp_df)
        predictions = predictions.reshape(512, 512)
        predictions = (predictions > 0.5).astype(float)
        cv2.imwrite(f'/content/{image_name[0]}.png', 255 * predictions)
