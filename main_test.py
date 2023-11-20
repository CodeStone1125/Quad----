import pytest
from PIL import Image
import cv2
import numpy as np
from collections import Counter
import sys
sys.path.append('build')
import quad  # Import your pybind11 module
sys.path.pop()

MODE_RECTANGLE = 1
MODE_ELLIPSE = 2
MODE_ROUNDED_RECTANGLE = 3

MODE = MODE_ELLIPSE
ITERATIONS = 128
LEAF_SIZE = 4
PADDING = 1
FILL_COLOR = (0, 0, 0)
SAVE_FRAMES = False
ERROR_RATE = 0.5
AREA_POWER = 0.25
OUTPUT_SCALE = 1

def rounded_rectangle(draw, box, radius, color):
    l, t, r, b = box
    d = radius * 2
    draw.ellipse((l, t, l + d, t + d), color)
    draw.ellipse((r - d, t, r, t + d), color)
    draw.ellipse((l, b - d, l + d, b), color)
    draw.ellipse((r - d, b - d, r, b), color)
    d = radius
    draw.rectangle((l, t + d, r, b - d), color)
    draw.rectangle((l + d, t, r - d, b), color)

def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print('Usage: python main.py input_image')
        return
    print(args[0])
    model = quad.Model(args[0])
    previous = None
    for i in range(ITERATIONS):
        error = model.averageError()
        if previous is None or previous - error > ERROR_RATE:
            print(i, error)
            if SAVE_FRAMES:
                model.render('frames/%06d.png' % i)
            previous = error
        model.split()
    model.render('output.jpg')  # 假設 max_depth 為 0
    print('-' * 32)
    heap = model.getQuads()
    depth = Counter(x.m_depth for x in heap)
    for key in sorted(depth):
        value = depth[key]
        n = 4 ** key

        if n != 0:
            pct = 100.0 * value / n
            print('%3d %8d %8d %8.2f%%' % (key, n, value, pct))
        else:
            # 處理 n 為零的情況
            pct = 0
            print('%3d %8d %8d %8s' % (key, n, value, "N/A"))
            print('%3d %8d %8d %8.2f%%' % (key, n, value, pct))
    print('-' * 32)
    print('             %8d %8.2f%%' % (len(model.getQuads()), 100))
    for max_depth in range(max(depth) + 1):
        model.render('out%d.jpg' % max_depth)




if __name__ == '__main__':
    main()
