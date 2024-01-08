import torch
from torch import optim, nn
from torchvision.datasets import ImageFolder
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import tifffile


data_path = "./Testing_Data"


def train_imshow():
    classes = ('img',)  # Defining the classes we have
    dataiter = iter(train)
    images, labels = next(dataiter)
    for i in range(len(images)):
        print(images[i].shape)
        plt.imshow(images[i].permute(1, 2, 0).numpy())
        plt.title(' '.join('%5s' % classes[labels[i]]))
        plt.show()
        
train_transform = transforms.Compose([
    transforms.Resize((3345, 3393)),
    transforms.ToTensor()])

train_set = datasets.ImageFolder(cwd + data_path, transform=train_transform)

train_imshow()
