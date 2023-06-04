# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
#import torch
#import torchvision
#from torchvision import datasets, transforms


#checking for device
#device = torch.device('metal:{}'.format(torch.cuda.device_count()))

#print(device)


def extract_color_features(image, target_size):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the number of bins for each channel in the histogram
    hue_bins = 8
    saturation_bins = 8
    value_bins = 8

    # Calculate the color histogram for each channel
    hue_hist = cv2.calcHist([hsv_image], [0], None, [hue_bins], [0, 180])
    saturation_hist = cv2.calcHist([hsv_image], [1], None, [saturation_bins], [0, 256])
    value_hist = cv2.calcHist([hsv_image], [2], None, [value_bins], [0, 256])

    # Normalize the histograms
    cv2.normalize(hue_hist, hue_hist, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(saturation_hist, saturation_hist, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(value_hist, value_hist, 0, 1, cv2.NORM_MINMAX)

    # Concatenate the histograms into a single feature vector
    color_features = np.concatenate((hue_hist.flatten(), saturation_hist.flatten(), value_hist.flatten()))

    # Resize the color features to the target size
    if len(color_features) < target_size:
        color_features = np.pad(color_features, (0, target_size - len(color_features)), mode='constant')
    elif len(color_features) > target_size:
        color_features = color_features[:target_size]

    print("Color features:", color_features.shape)

    return color_features


def extract_shape_features(image, target_size):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to obtain a binary image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store shape features
    shape_features = []

    # Iterate over the contours
    for contour in contours:
        # Calculate contour-based features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        _, _, width, height = cv2.boundingRect(contour)
        aspect_ratio = width / float(height) if height != 0 else 0
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0

        # Append the shape features to the list
        shape_features.extend([area, perimeter, aspect_ratio, circularity])

    # Convert the shape features list to a numpy array
    shape_features = np.array(shape_features)

    # Resize the shape features to the target size
    if len(shape_features) < target_size:
        shape_features = np.pad(shape_features, (0, target_size - len(shape_features)), mode='constant')
    elif len(shape_features) > target_size:
        shape_features = shape_features[:target_size]

    print("Shape features:", shape_features.shape)

    return shape_features


def extract_texture_features(image, target_size):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the Local Binary Pattern (LBP) for the grayscale image
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')

    # Calculate the histogram of the LBP image
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    # Flatten and return the histogram as the texture feature vector
    texture_features = hist.flatten()

    # Resize the texture features to the target size
    if len(texture_features) < target_size:
        texture_features = np.pad(texture_features, (0, target_size - len(texture_features)), mode='constant')
    elif len(texture_features) > target_size:
        texture_features = texture_features[:target_size]

    print("Texture features:", texture_features.shape)

    return texture_features


def combine_features(image):
    # Load and preprocess the image
    # Assuming image is already loaded or you can use OpenCV to load it
    preprocessed_image = image

    # Extract features using different methods
    color_features = extract_color_features(preprocessed_image,24)
    shape_features = extract_shape_features(preprocessed_image,20)
    texture_features = extract_texture_features(preprocessed_image,10)

    # Combine the features into a single vector
    combined_features = np.concatenate((color_features, shape_features, texture_features))

    return combined_features


folder_path = "/Users/hadi/Desktop/Concordia/Comp 6721/AIproject/fruitImages/Banana"
file_list = os.listdir(folder_path)

def loadImages(folder_path):
    for file_name in file_list:
         if file_name.endswith(".jpg") or file_name.endswith(".png"):
             image_path = os.path.join(folder_path, file_name)
             # Perform your image processing tasks here
             image = cv2.imread(image_path)
             new_size = (32, 32)
             image = cv2.resize(image, new_size)
             combined_features = combine_features(image)
             print(combined_features)
             #image.close()
        

            

# Print the shape of the combined feature vector
loadImages(folder_path)
#a = np.random.rand(20000,100)
#print(a)



