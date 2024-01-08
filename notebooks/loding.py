from UNetModel import UNet 
import torch.nn as nn
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
import torchvision.transforms as T
import rasterio as rio
from rasterio.plot import show
from labels import *
from sklearn.model_selection import train_test_split
from PIL import Image
import tifffile as tiff
from patchify import patchify
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torchsummaryX import summary

scaler = MinMaxScaler()

train_directory = '/home/halas/Inzynierka/Data/Train'
train_mask_directory = '/home/halas/Inzynierka/Data/Masks'
patch_size = 256
orders_image = [] 
image_dataset = []  
for path, subdirs, files in os.walk(train_directory):

    images = os.listdir(path)  
    for i, image_name in enumerate(images):  
        if image_name.endswith(".tif"):  
            
            image = tiff.imread(path+"/"+image_name)
            image = image.transpose((1, 2, 0))
            SIZE_X = (image.shape[1]//patch_size)*patch_size
            SIZE_Y = (image.shape[0]//patch_size)*patch_size 
            image = image[:, :SIZE_Y, :SIZE_X]      
            print(image.shape)
            print("Now patchifying image:", path+"/"+image_name)
            orders_image.append(image_name)
            patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
    
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    
                    single_patch_img = patches_img[i,j,0,:,:]
                    min_value = np.min(single_patch_img)
                    max_value = np.max(single_patch_img)
                    scale_factor = 255 / (max_value - min_value)
    
                    converted_image = ((single_patch_img - min_value) * scale_factor).astype(np.int16)
                    converted_image = converted_image / 255.0
                    converted_image = converted_image.astype('float32')
                    image_dataset.append(converted_image)

image_dataset = np.array(image_dataset)
print(image_dataset.shape)                

mask_dataset = []  
for path, subdirs, files in os.walk(train_mask_directory):
    for i, mask_name in enumerate(orders_image):  
        mask_name = mask_name.replace('.tif', '.png')
        print(path+"/"+"mask_"+mask_name)
        mask = cv2.imread(path+"/"+"mask_"+mask_name, 1)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
        SIZE_X = (mask.shape[1]//patch_size)*patch_size 
        SIZE_Y = (mask.shape[0]//patch_size)*patch_size 
        mask = Image.fromarray(mask)
        mask = mask.crop((0 ,0, SIZE_X, SIZE_Y)) 
        mask = np.array(mask)             


        print("Now patchifying mask:", path+"/"+mask_name)
        patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)

        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                
                single_patch_mask = patches_mask[i,j,:,:]
                single_patch_mask = single_patch_mask[0]                           
                mask_dataset.append(single_patch_mask) 
 
mask_dataset =  np.array(mask_dataset)

# # ############################################################################


labels = []
for i in range(len(mask_dataset)):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)    

labels = np.array(labels)   
labels = labels.astype(np.int64)
# # ############################################################################

X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels, test_size = 0.25 , random_state = 0)

#image_number = random.randint(0, len(X_train))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(X_train[image_number])
# #plt.subplot(122)
# #plt.imshow(y_train[image_number])
# plt.show()

 

# #######################################

# Convert the data to PyTorch tensors and create DataLoader
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model, criterion, and optimizer
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

model = UNet(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, img_channels=IMG_CHANNELS)

# Move the model to the appropriate device (e.g., GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 50

train_losses = []
val_losses = []

best_val_loss = float('inf')
best_model_weights = None

for epoch in range(3):
    print("START")
    model.train()  # Set the model to training mode
    train_loss_epoch = 0.0
    
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        outputs = model(images)
        #masks = masks.to(torch.long)
        
        # Compute the loss
        loss = criterion(outputs, masks)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss_epoch += loss.item() * images.size(0)
    
    train_loss_epoch /= len(train_loader.dataset)
    train_losses.append(train_loss_epoch)

    # Validation loop after each epoch
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_loss_epoch = 0.0
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            #masks = masks.to(torch.long) 
            
            # Compute the loss
            loss = criterion(outputs, masks)
            val_loss_epoch += loss.item() * images.size(0)
            
        val_loss_epoch /= len(test_loader.dataset)
        val_losses.append(val_loss_epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {train_loss_epoch:.4f}, Validation Loss: {val_loss_epoch:.4f}")

        # Save the best model based on validation loss
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_model_weights = model.state_dict().copy()

print(train_losses)
print(val_losses)
# Load the best model weights
model.load_state_dict(best_model_weights)