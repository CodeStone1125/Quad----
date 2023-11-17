import sys
import cv2
import numpy as np

sys.path.append('build')
import quad

# sys.path.pop()  # 這個操作可能會影響之後的模塊導入，確保是有意為之的

# 讀取圖片
image_data = cv2.imread('assets/star.jpg')
img3 = quad.cropImage_test(image_data,(50, 300, 100, 100))

# 顯示圖片
cv2.imshow('Flipped Image', img3)
cv2.waitKey(0)