import numpy as np
import pandas as pd
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, perimeter
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.filters import roberts, sobel
from scipy.signal import wiener
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


# Define the folder where the dataset is stored
folder_dataset = os.path.join(os.getcwd(), 'dataset')
print(folder_dataset)

# Define the folder where the stretched images will be saved
folder_stretched = os.path.join(os.getcwd(), 'stretched_images')
if not os.path.exists(folder_stretched):
    os.makedirs(folder_stretched)

# Define the folder where the Wiener filtered images will be saved
folder_wiener = os.path.join(os.getcwd(), 'wiener_filtered_images')
if not os.path.exists(folder_wiener):
    os.makedirs(folder_wiener)

# Get a list of all the image files in the dataset folder
file_names = [os.path.join(folder_dataset, image)for image in os.listdir(folder_dataset)]

# Define a function to perform histogram stretching on the images


def histogram_streteched(file_names):
    img_stretch_list = []
    for filename in file_names:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img_stretched = cv2.equalizeHist(img)
        img_stretch_list.append(img_stretched)
        # Save the stretched image into the folder for stretched images
        cv2.imwrite(os.path.join(folder_stretched, os.path.basename(
            filename)[:-4]+'_stretched.jpg'), img_stretched)
    return img_stretch_list

# Define a function to perform Wiener filtering on the images


def wiener_filter(file_names):
    img_applied_wiener = []
    for filename in file_names:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img_wiener = wiener(img, (5, 5))
        img_applied_wiener.append(img_wiener)
        # Save the Wiener filtered image into the folder for Wiener filtered images
        cv2.imwrite(os.path.join(folder_wiener, os.path.basename(
            filename)[:-4]+'_wiener_filtered.jpg'), img_wiener)
    return img_applied_wiener


# Apply histogram stretching to the images
imgs_his_stretched = histogram_streteched(file_names)

# Apply Wiener filtering to the images
imgs_wiener_filtered = wiener_filter(file_names)
