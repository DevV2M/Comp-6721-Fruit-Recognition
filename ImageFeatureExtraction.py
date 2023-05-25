import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern



def extract_color_features(image):
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

    print("Color features: " , color_features.shape)

    return color_features

def extract_shape_features(image):
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

    print("Shape features: " , shape_features.shape)

    return shape_features

def extract_texture_features(image):
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

    print("Text features: " , texture_features.shape)

    return texture_features

def combine_features(image):
    # Load and preprocess the image
    # Assuming image is already loaded or you can use OpenCV to load it
    preprocessed_image = image

    # Extract features using different methods
    color_features = extract_color_features(preprocessed_image)
    shape_features = extract_shape_features(preprocessed_image)
    texture_features = extract_texture_features(preprocessed_image)

    # Combine the features into a single vector
    combined_features = np.concatenate((color_features, shape_features, texture_features))

    return combined_features


folder_path = "/Users/hadi/Desktop/Fruits/Orange"
file_list = os.listdir(folder_path)

def loadImages(folder_path):
    for file_name in file_list:
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_path = os.path.join(folder_path, file_name)
            # Perform your image processing tasks here
            image = cv2.imread(image_path)
            new_size = (7, 7)
            image = cv2.resize(image, new_size)
            combined_features = combine_features(image)
            print("Combined Features Shape:", combined_features.shape)
            #image.close()


# Print the shape of the combined feature vector
loadImages(folder_path)

