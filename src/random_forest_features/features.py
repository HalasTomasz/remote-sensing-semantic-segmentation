import cv2
import numpy as np
import pandas as pd
from skimage.filters import sobel, scharr, roberts, prewitt
from skimage.feature import canny
from scipy import ndimage as nd


# Extract the features using different gabor kernels
def addGabor(df, img):
    """Aplay gabor filters on image

    Args:
        df (pd.DataFrame): dataframe with data
        img (np.array): image that gabor filters will be applied on

    Returns:
        (pd.DataFrame): dataframe with input data and Gabor filters
    """

    num = 1
    kernels = []
    ksize = 9

    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
                    gabor_label = "gabor" + str(num)
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, cv2.CV_32F)
                    kernels.append(kernel)
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img
                    num += 1

    return df


# Extract the features using different edge detector methods
def addEdges(df, gray):
    """Aplay multiple edge filters on image

    Args:
        df (pd.DataFrame): dataframe with data
        img (np.array): image that edge filters will be applied on

    Returns:
        (pd.DataFrame): dataframe with input data and edge filters
    """

    canny_edges = canny(gray, 0.6)
    roberts_edges = roberts(gray)
    sobel_edges = sobel(gray)
    scharr_edges = scharr(gray)
    prewitt_edges = prewitt(gray)
    df['canny_edges'] = canny_edges.reshape(-1)
    df['roberts_edge'] = roberts_edges.reshape(-1)
    df['sobel_edge'] = sobel_edges.reshape(-1)
    df['scharr_edge'] = scharr_edges.reshape(-1)
    df['prewitt_edge'] = prewitt_edges.reshape(-1)

    return df


# Extract feutures using gaussian and median filters
def addFilter(df, gray):
    """Aplay median and gausian edge filters on image

    Args:
        df (pd.DataFrame): dataframe with data
        gray (np.array): 2D image

    Returns:
        (pd.DataFrame): dataframe with input data and filters
    """

    gaussian_3 = nd.gaussian_filter(gray, sigma=3)
    gaussian_7 = nd.gaussian_filter(gray, sigma=7)
    median_img = nd.median_filter(gray, size=3)
    df['gaussian_3'] = gaussian_3.reshape(-1)
    df['gaussian_3'] = gaussian_7.reshape(-1)
    df['gaussian_3'] = median_img.reshape(-1)

    return df


def calc_features(img):
    """Aply multiple filters on image

    Args:
        img (np.array): image

    Returns:
        (pd.DataFrame): dataframe with features
    """

    df = pd.DataFrame()
    img_1D = np.reshape(img, (-1))
    df['pixels'] = img_1D
    df = addGabor(df, img_1D)
    df = addEdges(df, img)
    df = addFilter(df, img)

    return df
