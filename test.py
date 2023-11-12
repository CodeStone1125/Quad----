import cv2 
import numpy as np
from PIL import Image

image_path = "/home/williechu1125/repo/QuadraCompress/assests/star.jpg"

def calculate_histogram_cv(image_path):
    # Load image using OpenCV
    im = cv2.imread(image_path)

    # Convert BGR to RGB
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Initialize an empty list to store histograms for each channel
    hist_cv_channels = []

    # Calculate histogram for each channel
    for channel in range(im_rgb.shape[2]):
        hist_channel = cv2.calcHist([im_rgb], [channel], None, [256], [0, 256])
        hist_cv_channels.append(hist_channel)

    # Concatenate the histograms for all channels
    hist_cv = np.concatenate(hist_cv_channels)

    # Flatten the histogram values to 1D and cast to integers
    hist_cv_flattened = np.ravel(hist_cv).astype(int)

    # Get the image size (width and height)
    height, width, _ = im.shape

    return hist_cv_flattened, (width, height)  # Return the flattened histogram and image size

def calculate_histogram_pillow(image_path):
    # Load image using Pillow
    im = Image.open(image_path).convert('RGB')

    # Calculate histogram using Pillow
    hist = im.histogram()

    return hist, im.size  # Return the flattened histogram and image size


def main():
    # 讀取圖像
    image_cv2 = cv2.imread(image_path)

    # Get the width and height of the image
    height, width, _ = image_cv2.shape

    print("Image Width:", width)
    print("Image Height:", height)

    # 將圖像轉換為RGB模式
    image_rgb_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    # 計算OpenCV直方圖
    hist_cv2, size_cv2 = calculate_histogram_cv(image_path)
    hist_cv2_list = hist_cv2.tolist()  # Convert to regular Python list

    # 使用Pillow計算直方圖
    hist_pillow, size_pillow = calculate_histogram_pillow(image_path)

    # 輸出兩者的直方圖和形狀信息
    print("OpenCV 直方圖：", hist_cv2_list)
    print("OpenCV 直方圖形狀：", size_cv2)
    print("Pillow 直方圖：", hist_pillow)
    print("Pillow 直方圖形狀：", size_pillow)

    # 比較兩者直方圖是否相同
    if np.array_equal(hist_cv2, hist_pillow) and size_cv2 == size_pillow:
        print("兩者的直方圖相同")
    else:
        print("兩者的直方圖不同")

if __name__ == "__main__":
    main()

