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
ITERATIONS = 128
LEAF_SIZE = 4
PADDING = 1
FILL_COLOR = (0, 0, 0)
SAVE_FRAMES = False
ERROR_RATE = 0.5
AREA_POWER = 0.25
OUTPUT_SCALE = 1

# Global variable for model


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

def render(model, path, max_depth=10):
    # print("into render")
    m = OUTPUT_SCALE
    dx, dy = (PADDING, PADDING)
    im = Image.new('RGB', (model.width * m + dx, model.height * m + dy))
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, model.width * m, model.height * m), FILL_COLOR)
    
    frames_folder = 'frames'  # Specify the frames folder
    root = model.root
    # print(root.color)
    # print(root.depth)
    # print(root.children)
    for i, quad in enumerate(root.get_leaf_nodes(max_depth)):
        # print("quad:", i)
        x, y, width, height = quad.box
        box = (x * m + dx, (y + height) * m + dy, (x + width) * m - 1, y * m - 1)
        # print("box:", box)
        # print("color:", quad.color)
        # print(MODE)
        if MODE == MODE_ELLIPSE:
            draw.ellipse(box, quad.color)
        elif MODE == MODE_ROUNDED_RECTANGLE:
            radius = m * min(width, height) / 4
            rounded_rectangle(draw, box, radius, quad.color)
        else:
            draw.rectangle(box, quad.color)
        
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
    print("------------------childrem for quad:" ,quad.children)
    for child in quad.children:
        model.push(child)
        print(f"children color: ({child.color[0]}, {child.color[1]}, {child.color[2]})")
        model.error_sum += child.error * child.area

def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print('Usage: python main.py input_image')
        return
    print(args[0])
    
    model = quad.Model(args[0])
    # print("------------------------test model-----------------------------------------")
    # print(model.width) #640
    # print(model.height) #487
    # print(model.root.color) #(125, 147, 146)
    # print(model.root.error) #55.80494445223306
    # print(model.root.area) #311680.0
    # print(model.error_sum) #17393285.086872
    # print(model.root.children)
    # print("------------------------test end-----------------------------------------")
#-------------------------------------OK--------------------------------------#
    previous = None
    error = model.averageError()
    if previous is None or previous - error > ERROR_RATE:
        if SAVE_FRAMES:
            render(model,'frames/%06d.png' % i)
        previous = error
    #quadtree = model.pop()
    _children = model.root.split()
    for child in _children:
        model.push(child)
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
    for max_depth in range(max(depth) + 1):
       render( model, 'out%d.png' % max_depth)




if __name__ == '__main__':
    main()
