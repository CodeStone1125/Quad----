import sys
import cv2
import numpy as np

sys.path.append('build')



# 創建一個示例圖像數據
image_data_2 = np.random.randint(0, 256, size=(300, 400, 3), dtype=np.uint8)

# 查看數據類型

print(type(image_data_2))