import cv2
import numpy as np
from PIL import Image

def calculate_histogram_cv(rgb_image):
    # Convert the RGB image to a NumPy array
    np_image = np.array(rgb_image)

    # Initialize a list to store histograms for each channel
    hist_cv_channels = []

    # Calculate histogram for each channel
    for channel in range(np_image.shape[2]):
        hist_channel = cv2.calcHist([np_image], [channel], None, [256], [0, 256])
        hist_cv_channels.extend(hist_channel.flatten())

    # Cast the values to integers
    hist_cv_channels = [int(value) for value in hist_cv_channels]

    return hist_cv_channels

def calculate_histogram_pillow(image_path):
    # Load image using Pillow
    im = Image.open(image_path).convert('RGB')

    # Calculate histogram using Pillow
    hist = im.histogram()

    # Return the histogram as a list of integers
    return [int(value) for value in hist]

# Load image using Pillow
image_path = "/home/williechu1125/repo/QuadraCompress/assests/star.jpg"
im = Image.open(image_path).convert('RGB')

# Calculate histograms using both functions
histogram_cv = calculate_histogram_cv(im)
histogram_pillow = calculate_histogram_pillow(image_path)

# Compare histograms
if histogram_cv == histogram_pillow:
    print("Histograms are equal.")
else:
    print("Histograms are not equal.")
