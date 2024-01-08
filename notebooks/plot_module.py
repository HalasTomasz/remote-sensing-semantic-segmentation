import os
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from PIL import Image

def add_text(ax, x, y, txt, idx, space=1, color='black'):
    text_height = 0.02
    offset = text_height + space
    if idx == 1:
         ax.text(x, y + space, txt, color=color, ha='center', va='bottom')
    elif  idx == 0:
        ax.text(x, y - text_height - space, txt, color=color, ha='center', va='top')
    else:
        if y % (2 * offset) < offset:
            ax.text(x, y + space, txt, color=color, ha='center', va='bottom')
        else:
            ax.text(x, y - text_height - space, txt, color=color, ha='center', va='top')



def make_metrics_plot(df, model):
    epoch = [i for i in range(len(df))]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epoch, df['Accuracy'], marker='o', color='blue')
    for i, txt in enumerate(df['Accuracy']):
        add_text(ax, epoch[i], txt, f'{txt:.2f}', i)
    ax.set_xlabel("Number of Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.5, 110)
    ax.grid(linestyle='--')
    fig.savefig(f'accuracy_{model}_plot.png',  dpi=300)
    

    # Plot Precision
    ax.plot(epoch, df['Precision'], marker='o', color='blue')
    for i, txt in enumerate(df['Precision']):
        add_text(ax, epoch[i], txt, f'{txt:.2f}', i, space=0.01)
    ax.set_xlabel("Number of Epoch")
    ax.set_ylabel("Precision")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(linestyle='--')
    fig.savefig(f'precision_{model}_plot.png',  dpi=300)

    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot Recall
    ax.plot(epoch, df['Recall'], marker='o', color='blue')
    for i, txt in enumerate(df['Recall']):
        add_text(ax, epoch[i], txt, f'{txt:.2f}', i, space=0.01)
    ax.set_xlabel("Number of Epoch")
    ax.set_ylabel("Recall")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(linestyle='--')
    fig.savefig(f'recall_{model}_plot.png',  dpi=300)

    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot F1 Score
    ax.plot(epoch, df['F1 Score'], marker='o', color='blue')
    for i, txt in enumerate(df['F1 Score']):
        add_text(ax, epoch[i], txt, f'{txt:.2f}', i, space=0.01)
    ax.set_xlabel("Number of Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(linestyle='--')
    fig.savefig(f'f1_score_{model}_plot.png',  dpi=300)


UNET_FIle ='/home/halas/UNET_RESULTS'
RESNET_FILE = '/home/halas/RESNET_REULTS'
RESULT_FILE= '/home/halas/RESULTS'
RESULT_FILE_TEST= '/home/halas/AerialImageDatasetPATCH/test/images'
pattern = re.compile(r"acc (\d+\.\d+).*?Dice score: (\d+\.\d+).*?Precision: (\d+\.\d+).*?Recall: (\d+\.\d+).*?F1 Score: (\d+\.\d+)", re.DOTALL)

with open (UNET_FIle+'/result.txt') as f:
    data = f.read()

matches = pattern.findall(data)

df = pd.DataFrame(matches, columns=['Accuracy', 'Dice', 'Precision', 'Recall', 'F1 Score'], dtype=float)

make_metrics_plot(df, "UNET")

# def plot_images_from_folders(folder_with_subfolders, folder_with_images):
#     # Get the list of subfolders in the first directory
#     subfolders = [f.path for f in os.scandir(folder_with_subfolders) if f.is_dir()]

#     # Get the list of images in the second directory
#     images = sorted([f.path for f in os.scandir(folder_with_images) if f.is_file() and f.name.lower().endswith(('.tif'))])

#     # Create a 3x3 subplot grid
#     fig, axes = plt.subplots(3, 4, figsize=(10, 10))
#     axes = axes.flatten()
#     subfolder_images = []
#     # Plot images from subfolders
#     for i in range(3):
#         subfolder_path = subfolders[i]
#         subfolder_images.append(sorted([f.path for f in os.scandir(subfolder_path) if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg'))]))
    
#     labels = ['Orignal', "RESUNET-50", 'U-NET', 'RANDOM FOREST']
#     for column, (img, resnet_img, unet_img, forest_img) in enumerate(zip(images, subfolder_images[0], subfolder_images[1], subfolder_images[2])):

#         img_list = [img, resnet_img, unet_img, forest_img]

#         for row, (img_path, label) in enumerate(zip(img_list, labels)):

#             filename_without_extension = os.path.splitext(os.path.basename(img_path))[0]
          
#             name_parts = filename_without_extension.split('_')
#             dataset_name = name_parts[0].upper()
#             patch_number = name_parts[-1].upper() 

#             if row == 0:
#                 axes[column % 3 * 4 + row].set_title(dataset_name + " PATCH " + patch_number, fontsize=14)
#             else:
#                 axes[column % 3 * 4 + row].set_title(label, fontsize=14)

#             if label == "RANDOM FOREST":
#                 img = Image.open(img_path)
#                 img_array = np.array(img)
#                 img_array = (img_array > 128).astype(np.float32)
#                 img_array = img_array * 255
#                 img = Image.fromarray(img_array)
#                 axes[column % 3 * 4 + row].imshow(img)
#                 axes[column % 3 * 4 + row].axis('off')

#             else:
#                 img = Image.open(img_path)
#                 axes[column % 3 * 4 + row].imshow(img)
#                 axes[column % 3 * 4 + row].axis('off')
#         if (column + 1) % 3 == 0:
#             fig.tight_layout()
#             fig.subplots_adjust(wspace=0.1, hspace=0.1)
#             #fig.show()
#             fig.savefig(f'img/res/{column}_idx_plot.png',  dpi=300)

#             fig, axes = plt.subplots(3, 4, figsize=(10, 10))
#             axes = axes.flatten()


# plot_images_from_folders(RESULT_FILE, RESULT_FILE_TEST)