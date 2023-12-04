import time
import psutil
from tkinter import filedialog
from tkinter import Text
from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import StringVar
from PIL import Image, ImageDraw
from PIL import Image
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
    start_time = time.time()  # 開始時間

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
    end_time = time.time()  # 結束時間
    elapsed_time = end_time - start_time  # 實際時間


    print(f"elapsed_time：{elapsed_time:.2f} 秒")
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
        self.priv_card.pack(side=TOP, fill=BOTH, padx=(10, 5), pady=10)


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
            padding=40,
        )
        note_frame.pack(fill=BOTH)

        # # progressbar with text indicator
        # pb_frame = ttk.Frame(note_frame, padding=(0, 10, 10, 10))  # 注意這裡修改成 note_frame
        # pb_frame.pack(side=TOP, fill=X, expand=YES)

        # pb = ttk.Progressbar(
        #     master=pb_frame,
        #     bootstyle=(SUCCESS, STRIPED),
        #     variable='progress'
        # )
        # pb.pack(side=LEFT, fill=X, expand=YES, padx=(15, 10))

        # ttk.Label(pb_frame, text='%').pack(side=RIGHT)
        # ttk.Label(pb_frame, textvariable='progress').pack(side=RIGHT)
        # self.setvar('progress', 78)

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
        wt_canvas.pack(side=LEFT, fill=BOTH, ipadx=10)  # 設定 ipadx 以進行內部填充

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

        # static frame in the main frame
        static_frame = ttk.Labelframe(
            master=scroll_frame,
            text='STATIC',
            padding=(20, 5)
        )
        static_frame.pack(side=BOTTOM, fill=BOTH, padx=20, pady=10, expand=YES)

        # text element in the static frame
        self.text_widget = Text(
            master=static_frame,
            wrap='word',  # 控制文本如何換行
            width=30,  # 設定寬度
            height=18,  # 設定高度
            state='disabled'  # 設定為禁用，無法編輯
        )
        self.text_widget.pack(fill=BOTH, expand=YES)


        # 將 sys.stdout 重新導向到 Text 元件
        sys.stdout = TextRedirector(self.text_widget, "stdout")

        
        # Input and Set buttons
        edge = ttk.Labelframe(
            master=scroll_frame,
            text='ITERATIONS',
            padding=(20, 5)
        )
        edge.pack(side=TOP, fill=BOTH, expand=YES, padx=20, pady=10)

        # Entry for input
        self.input_entry = ttk.Entry(edge, bootstyle=PRIMARY)
        self.input_entry.insert(0, "1024")  # 將值設置為 1024
        self.input_entry.pack(side=LEFT, padx=5)

        # Set button
        set_button = ttk.Button(edge, text='Set', command=self.set_iterations)
        set_button.pack(side=LEFT, padx=5)


        explorer = ttk.Labelframe(
            master=scroll_frame,
            text='DRAW_MODE',
            padding=(20, 5)
        )
        explorer.pack(side=TOP, fill=BOTH, padx=20, pady=10, expand=YES)

        # 使用 StringVar 來控制選擇
        self.selected_draw_mode = StringVar()

        # add radio buttons to each label frame section
        for opt in radio_options:
            rb = ttk.Radiobutton(explorer, text=opt, variable=self.selected_draw_mode, value=opt, command=self.update_mode)
            rb.pack(side=TOP, pady=2, fill=X)
                
            if opt == 'MODE_RECTANGLE':
                self.selected_draw_mode.set(opt)  # 設定變數為 'MODE_RECTANGLE'
            else:
                rb.configure(state='disabled')  # 禁用其他選項
    
        notebook.add(windows_tab, text='windows')

        # empty tab for looks
        notebook.add(ttk.Frame(notebook), text='about me')

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("All files", "*.*")]
        )

        if file_path:
            # 清除先前的圖片
            for widget in self.priv_card.winfo_children():
                widget.destroy()

            # 處理圖片的程式碼
            original_image = Image.open(file_path)

            # 設定自訂的圖片顯示尺寸
            desired_width = 850  # 自訂寬度
            desired_height = 500  # 自訂高度

            # 調整圖片大小以符合自訂的寬度和高度
            original_image.thumbnail((desired_width, desired_height))

            original_image = ImageTk.PhotoImage(original_image)

            # 在 result card 中顯示原圖
            original_image_label = ttk.Label(master=self.priv_card, image=original_image)
            original_image_label.image = original_image
            original_image_label.pack(fill=BOTH, expand=YES)

            # 延遲顯示處理後的圖片
            self.after(2000, lambda: self.show_processed_image(file_path))

    def show_processed_image(self, file_path):
        # 清除先前的圖片
        for widget in self.priv_card.winfo_children():
            widget.destroy()
        process_image_with_model(file_path)

        processed_image = Image.open("./output.jpg")

        # 計算 cards_frame 的寬度和高度
        frame_width = self.priv_card.winfo_width()
        frame_height = self.priv_card.winfo_height()

        # 調整圖片大小以符合 cards_frame 的寬度和高度
        processed_image.thumbnail((frame_width, frame_height))

        processed_image = ImageTk.PhotoImage(processed_image)

        # 在 result card 中顯示處理後的圖片
        processed_image_label = ttk.Label(master=self.priv_card, image=processed_image)
        processed_image_label.image = processed_image
        processed_image_label.pack(fill=BOTH, expand=YES)

# def update_mode(selected_draw_mode):
#     mode_str = selected_draw_mode
    
#     if mode_str == 'MODE_RECTANGLE':
#         MODE = MODE_RECTANGLE
#     elif mode_str == 'MODE_ELLIPSE':
#         MODE = MODE_ELLIPSE
#     elif mode_str == 'MODE_ROUNDED_RECTANGLE':
#         MODE = MODE_ROUNDED_RECTANGLE

    #     print("Selected Mode:", MODE)
    def update_mode(self):
        mode_str = self.selected_draw_mode.get()
        if mode_str == 'MODE_RECTANGLE':
            MODE = MODE_RECTANGLE
        elif mode_str == 'MODE_ELLIPSE':
            MODE = MODE_ELLIPSE
        elif mode_str == 'MODE_ROUNDED_RECTANGLE':
            MODE = MODE_ROUNDED_RECTANGLE
        print("Selected Mode:", MODE)

    def set_iterations(self):
        global ITERATIONS
        try:
            ITERATIONS = int(self.input_entry.get())
            print("ITERATIONS set to:", ITERATIONS)
        except ValueError:
            print("Invalid input for ITERATIONS")




# TextRedirector 類別用於將 stdout 重定向到 Tkinter Text 元件
class TextRedirector:
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.config(state='normal')
        #
        self.widget.insert("end", str, (self.tag,))  # 再插入新文本
        self.widget.update()  # 立即更新
        self.widget.config(state='disabled')
        self.widget.see("end")
    def flush(self):
        pass  # 簡單的 pass，因為我們不需要實際的 flush 操作



if __name__ == '__main__':

    app = ttk.Window("PC Cleaner", "simplex", resizable=(False, False))
    app.geometry("1280x720")  # Set window size
    Cleaner(app)
    app.mainloop()
