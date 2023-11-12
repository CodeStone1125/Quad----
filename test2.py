import sys
sys.path.append('build')

import example
import cv2
from PIL import Image
import numpy as np

# Load image using Pillow
image_path = "/home/williechu1125/repo/QuadraCompress/assests/star.jpg"
im = Image.open(image_path).convert('RGB')

# Convert Pillow image to NumPy array
np_image = np.array(im)

# Convert NumPy array to cv::Mat
cv_image = cv2.UMat(np_image)

# Call the C++ function
histogram_cpp = example.calculate_histogram_cv(cv_image)

def calculate_histogram_pillow(image_path):
    # Load image using Pillow
    im = Image.open(image_path).convert('RGB')

    # Calculate histogram using Pillow
    hist = im.histogram()

    # Return the histogram as a list of integers
    return hist

histogram_pillow = calculate_histogram_pillow(image_path)

# Compare histograms
if histogram_cpp == histogram_pillow:
    print("Histograms are equal.")
else:
    print("Histograms are not equal.")
