from PIL import Image, ImageDraw
from PIL import Image
# import cv2
# import numpy as np
from collections import Counter
import sys
sys.path.append('build')
import quad  # Import your pybind11 module
sys.path.pop()

MODE_RECTANGLE = 1
MODE_ELLIPSE = 2
MODE_ROUNDED_RECTANGLE = 3

MODE = MODE_RECTANGLE
ITERATIONS = 512
LEAF_SIZE = 4
PADDING = 1
FILL_COLOR = (0, 0, 0)
SAVE_FRAMES = False
ERROR_RATE = 0.5
AREA_POWER = 0.25
OUTPUT_SCALE = 1

# Global variable for model


def rounded_rectangle(draw, box, radius, color):
    x, y, width, height = box
    l = x
    t = y 
    r = x + width
    b = y + height
    d = radius * 2
    draw.ellipse((l, t, l + d, t + d), color)
    draw.ellipse((r - d, t, r, t + d), color)
    draw.ellipse((l, b - d, l + d, b), color)
    draw.ellipse((r - d, b - d, r, b), color)
    d = radius
    draw.rectangle((l, t + d, r, b - d), color)
    draw.rectangle((l + d, t, r - d, b), color)

def render(model, path, max_depth=None):
    # print("into render")
    m = OUTPUT_SCALE
    dx, dy = (PADDING, PADDING)
    im = Image.new('RGB', (model.width * m + dx, model.height * m + dy))
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, model.width * m, model.height * m), FILL_COLOR)

    frames_folder = 'frames'  # Specify the frames folder
    for i, quad in enumerate(get_leaf_nodes(model, max_depth)):

        x, y, width, height = quad.box
        box = (x * m + dx, (y + height) * m + dy, (x + width) * m - 1, y * m - 1)

        # 將 BGR 轉換成 RGB
        color_rgb = (quad.color[2], quad.color[1], quad.color[0])

        if MODE == MODE_ELLIPSE:
            draw.ellipse(box, color_rgb)
        elif MODE == MODE_ROUNDED_RECTANGLE:
            radius = m * min(width, height) / 4
            rounded_rectangle(draw, box, radius, color_rgb)
        else:
            draw.rectangle(box, color_rgb)

        # Save each frame into the "frames" folder
        frame_path = f"{frames_folder}/out{i:03d}.png"
        im.save(frame_path, 'PNG')

    del draw
    im.save(path, 'PNG')

def split(model):
    quad = model.pop()
    model.error_sum -= quad.error * quad.area
    _children = quad.split()
    quad.children = _children
    # print("------------------childrem for quad:" ,quad.children)
    for child in quad.children:
        model.push(child)
        # print(f"children color: ({child.color[0]}, {child.color[1]}, {child.color[2]})")
        model.error_sum += child.error * child.area

def get_leaf_nodes(model, max_depth=None):
    # print("into get_leaf_nodes")
    leaves = []
    heap = model.getQuads()
    for quad in heap:
        if not quad.children:
            leaves.append(quad)
        if max_depth is not None and quad.depth >= max_depth:
            leaves.append(quad)
        else:
            # 在這裡添加額外的邏輯，如果有的話
            pass
    
    return leaves


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print('Usage: python main.py input_image')
        return
    print(args[0])
    
    model = quad.Model(args[0])
    previous = None
    error = model.averageError()
    if previous is None or previous - error > ERROR_RATE:
        if SAVE_FRAMES:
            render(model,'frames/%06d.png' % i)
        previous = error
    _children = model.root.split()
    for child in _children:
        model.push(child)
        #_children = child.split()
        print(f"children color: ({child.color[0]}, {child.color[1]}, {child.color[2]})")
        model.error_sum += child.error * child.area
    render(model,'output.jpg')
    for i in range(ITERATIONS-1):
        error = model.averageError()
        if previous is None or previous - error > ERROR_RATE:
            print(i, error)
            if SAVE_FRAMES:
                render(model,'frames/%06d.png' % i)
            previous = error
        split(model)
    render(model,'output.jpg')
    print('-' * 32)
    heap = model.getQuads()
    print("model heap size:", len(heap))
    depth = Counter(x.depth for x in heap)
    for key in sorted(depth):
        value = depth[key]
        n = 4 ** key
        pct = 100.0 * value / n
        print('%3d %8d %8d %8.2f%%' % (key, n, value, pct))
    print('-' * 32)
    print('             %8d %8.2f%%' % (len(model.getQuads()), 100))
    print(max(depth))
    # for max_depth in range(max(depth) + 1):
    #    render( model, 'out%d.png' % max_depth)




if __name__ == '__main__':
    main()
