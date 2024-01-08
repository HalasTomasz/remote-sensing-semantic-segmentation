import numpy as np

color_dict = {
    1: [0, 128, 0], # Green
    2: [255, 0, 0], # Red
    3: [0, 0, 255], # Blue
    4: [255, 255, 0], # Yellow
    5: [255, 192, 203], # Pink
    6: [128, 0, 128], # Purple
    7: [255, 165, 0], # Orange
    8: [0, 255, 0], # Lime Green
    9: [0, 0, 0], # Black
    10: [255, 255, 255] # White
}

BUILDING = np.array([0, 128, 0])
SMALL_STRUCTURE= np.array([255, 0, 0])
ROADS = np.array([0, 0, 255]),
WOODLAND = np.array([255, 255, 0]),
NON_RESIDENTIAL_BUILDING= np.array([255, 192, 203])
CROPLAND = np.array([128, 0, 128])
WATERWAY = np.array([255, 165, 0])
STANDING_WATER = np.array([0, 255, 0])
LARGE_VEHICLE = np.array([0, 0, 0])
MOTORBIKE = np.array([255, 255, 255]),

def rgb_to_2D_label(label):
    label_seg = np.zeros(label.shape,dtype=np.int16)
    label_seg[np.all(label == BUILDING,axis=-1)] = 0
    label_seg[np.all(label == SMALL_STRUCTURE,axis=-1)] = 1
    label_seg[np.all(label == ROADS,axis=-1)] = 2
    label_seg[np.all(label == WOODLAND,axis=-1)] = 3
    label_seg[np.all(label == NON_RESIDENTIAL_BUILDING,axis=-1)] = 4
    label_seg[np.all(label == CROPLAND,axis=-1)] = 5
    label_seg[np.all(label == WATERWAY,axis=-1)] = 6
    label_seg[np.all(label == STANDING_WATER,axis=-1)] = 7
    label_seg[np.all(label == LARGE_VEHICLE,axis=-1)] = 8
    label_seg[np.all(label == MOTORBIKE,axis=-1)] = 9
    
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg