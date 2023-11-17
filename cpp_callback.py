import sys
import cv2
import numpy as np

sys.path.append('build')
import quad

# 讀取圖片並指定數據類型為 uint8
image_data = cv2.imread('assets/star.jpg').astype(np.uint8)

# 調用 C++ 回調函式
quad.cpp_callback1(True, "example_id", image_data)
