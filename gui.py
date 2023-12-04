from tkinter import filedialog
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import StringVar
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
ITERATIONS = 1024
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
        # frame_path = f"{frames_folder}/out{i:03d}.png"
        # im.save(frame_path, 'PNG')

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


def process_image_with_model(image_path):
    args = [image_path]
    model = quad.Model(args[0])
    previous = None
    error = model.averageError()
    if previous is None or previous - error > ERROR_RATE:
        if SAVE_FRAMES:
            render(model, 'frames/%06d.png' % i)
        previous = error
    _children = model.root.split()
    for child in _children:
        model.push(child)
        model.error_sum += child.error * child.area
    for i in range(ITERATIONS - 1):
        error = model.averageError()
        if previous is None or previous - error > ERROR_RATE:
            print(i, error)
            previous = error
        split(model)
    render(model, 'output.jpg')
    print('-' * 32)
    heap = model.getQuads()
    depth = Counter(x.depth for x in heap)
    for key in sorted(depth):
        value = depth[key]
        n = 4 ** key
        pct = 100.0 * value / n
        print('%3d %8d %8d %8.2f%%' % (key, n, value, pct))
    print('-' * 32)
    print('             %8d %8.2f%%' % (len(model.getQuads()), 100))


class Cleaner(ttk.Frame):

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.pack(fill=BOTH, expand=YES)

        # header
        hdr_frame = ttk.Frame(self, padding=20, bootstyle=INFO)
        hdr_frame.grid(row=0, column=0, columnspan=3, sticky=EW)

        # 設定列的權重，使hdr_frame寬度與視窗一致
        self.columnconfigure(0, weight=1)

        hdr_label = ttk.Label(
            master=hdr_frame,
            text='Quadra compressor',
            font=('TkDefaultFixed', 30),
            bootstyle=(INVERSE, INFO)
        )
        hdr_label.pack(side=LEFT, padx=10)

        # results frame
        results_frame = ttk.Frame(self)
        results_frame.grid(row=1, column=2, sticky=NSEW)

        # 設定列的權重，使results_frame寬度與視窗一致
        self.columnconfigure(2, weight=4)
        self.rowconfigure(1, weight=1)

        # result cards
        cards_frame = ttk.Frame(
            master=results_frame,
            name='cards-frame',
            bootstyle=LIGHT
        )
        cards_frame.pack(fill=BOTH, expand=YES)
        

        # 在 result card 中加入一個 frame 以放置選擇檔案載入圖片的按鈕
        action_frame = ttk.Frame(cards_frame, padding=20)
        action_frame.pack(side=TOP, fill=X)
        
        # 定義 priv_card
        self.priv_card = ttk.Frame(
            master=cards_frame,
            padding=1,
        )
        self.priv_card.pack(side=LEFT, fill=BOTH, padx=(10, 5), pady=10)


        # 載入圖片的按鈕
        load_image_btn = ttk.Button(
            master=action_frame,
            text='Load Image',
            command=self.load_image
        )
        load_image_btn.pack(side=TOP, fill=BOTH, ipadx=10, ipady=10)

        # user notification
        note_frame = ttk.Frame(
            master=results_frame, 
            bootstyle=LIGHT, 
            padding=40
        )
        note_frame.pack(fill=BOTH)

        # progressbar with text indicator
        pb_frame = ttk.Frame(note_frame, padding=(0, 10, 10, 10))  # 注意這裡修改成 note_frame
        pb_frame.pack(side=TOP, fill=X, expand=YES)

        pb = ttk.Progressbar(
            master=pb_frame,
            bootstyle=(SUCCESS, STRIPED),
            variable='progress'
        )
        pb.pack(side=LEFT, fill=X, expand=YES, padx=(15, 10))

        ttk.Label(pb_frame, text='%').pack(side=RIGHT)
        ttk.Label(pb_frame, textvariable='progress').pack(side=RIGHT)
        self.setvar('progress', 78)

        # option notebook
        notebook = ttk.Notebook(self)
        notebook.grid(row=1, column=0, sticky=NSEW, pady=(25, 0))  # 將column改為0

        # windows tab
        windows_tab = ttk.Frame(notebook, padding=10)
        wt_canvas = ttk.Canvas(
            master=windows_tab,
            relief=FLAT,
            borderwidth=0,
            selectborderwidth=0,
            highlightthickness=0,
        )
        wt_canvas.pack(side=LEFT, fill=BOTH)

        # adjust the scrollregion when the size of the canvas changes
        wt_canvas.bind(
            sequence='<Configure>',
            func=lambda e: wt_canvas.configure(
                scrollregion=wt_canvas.bbox(ALL))
        )

        scroll_frame = ttk.Frame(wt_canvas)
        wt_canvas.create_window((0, 0), window=scroll_frame, anchor=NW)

        radio_options = [
            'MODE_RECTANGLE', 'MODE_ELLIPSE', 'MODE_ROUNDED_RECTANGLE'
        ]

        # Input and Set buttons
        edge = ttk.Labelframe(
            master=scroll_frame,
            text='ITERATIONS',
            padding=(20, 5)
        )
        edge.pack(fill=BOTH, expand=YES, padx=20, pady=10)

        # Entry for input
        input_entry = ttk.Entry(edge, bootstyle=PRIMARY)
        input_entry.pack(side=LEFT, padx=5)

        # Set button
        set_button = ttk.Button(edge, text='Set')
        set_button.pack(side=LEFT, padx=5)

        # 使用 StringVar 來控制選擇
        selected_draw_mode = StringVar()

        explorer = ttk.Labelframe(
            master=scroll_frame,
            text='DRAW_MODE',
            padding=(20, 5)
        )
        explorer.pack(fill=BOTH, padx=20, pady=10, expand=YES)

        # add radio buttons to each label frame section
        for opt in radio_options:
            rb = ttk.Radiobutton(explorer, text=opt, variable=selected_draw_mode, value=opt)
            rb.pack(side=TOP, pady=2, fill=X)

        notebook.add(windows_tab, text='windows')

        # empty tab for looks
        notebook.add(ttk.Frame(notebook), text='about me')

    def load_image(self):
        # UNDO: try to change to display only image options not all option
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("All files", "*.*")]
        )

        if file_path:
            # 處理圖片的程式碼
            process_image_with_model(file_path)
            image = Image.open("./output.jpg")
            image = ImageTk.PhotoImage(image)

            # 在 result card 中顯示圖片
            image_label = ttk.Label(master=self.priv_card, image=image)
            image_label.image = image  # 保留對圖片對象的引用，以防止被垃圾回收
            image_label.pack(fill=BOTH, expand=YES)






if __name__ == '__main__':

    app = ttk.Window("PC Cleaner", "simplex", resizable=(False, False))
    app.geometry("1280x720")  # Set window size
    Cleaner(app)
    app.mainloop()
