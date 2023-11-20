import sys
import cv2
import numpy as np

sys.path.append('build')
import quad

# Python version histogram
def calculate_histogram_py(image):
    # Convert the image to numpy array if it's not
    image_array = np.asarray(image)

    # Calculate histogram using im.histogram()
    hist_py = np.histogram(image_array, bins=256, range=(0, 256))[0]

    return hist_py.tolist()

# Load an image using OpenCV
image_path = "assets/star.jpg"  # 替換成你的圖像文件路徑
image = cv2.imread(image_path)

# Calculate histograms using both Python and C++ versions
hist_py = calculate_histogram_py(image)
hist_cv = quad.calculate_histogram_cv(image)

# Print and compare the results
print("Python version histogram:")
print(hist_py)

print("\nC++ version histogram:")
print(hist_cv)
