import cv2 
import numpy as np
from PIL import Image

image_path = "/home/williechu1125/repo/QuadraCompress/assests/star.jpg"
image_cv2 = cv2.imread(image_path)
# 將圖像轉換為RGB模式
image_rgb_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def calculate_histogram_cv(rgb_image):
    # Initialize an empty list to store histograms for each channel
    hist_cv_channels = []

    # Calculate histogram for each channel
    for channel in range(rgb_image.shape[2]):
        hist_channel = cv2.calcHist([rgb_image], [channel], None, [256], [0, 256])
        hist_cv_channels.append(hist_channel)

    # Concatenate the histograms for all channels
    hist_cv = np.concatenate(hist_cv_channels)

    # Flatten the histogram values to 1D and cast to integers
    hist_cv_flattened = np.ravel(hist_cv).astype(int)

    return hist_cv_flattened

def calculate_histogram_pillow(image_path):
    # Load image using Pillow
    im = Image.open(image_path).convert('RGB')

    # Calculate histogram using Pillow
    hist = im.histogram()

    return hist

# Calculate histogram using the calculate_histogram_cv function
histogram_cv = calculate_histogram_cv(image_rgb_cv2)

# Calculate histogram using the calculate_histogram_pillow function
histogram_pillow = calculate_histogram_pillow(image_path)

# Print the histograms
print("Histogram (OpenCV):", histogram_cv)
print("Histogram (Pillow):", histogram_pillow)

# Compare whether the two histograms are equal
if np.array_equal(histogram_cv, histogram_pillow):
    print("Histograms are equal")
else:
    print("Histograms are not equal")
