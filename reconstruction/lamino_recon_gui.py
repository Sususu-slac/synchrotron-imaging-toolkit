# app.py
import os
import subprocess
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk

import h5py
import numpy as np
from PIL import Image, ImageTk

import recon_utilities as ru


# ===== Global GUI state =====
current_slice = 0
projections = None
slice_slider = None

# These GUI widgets are initialized in main().
input_path_entry = None
output_path_entry = None
log_text = None
image_label = None

rc_entry = None
css_entry = None
csw_entry = None
ta_entry = None
ast_entry = None
asw_entry = None
b_entry = None
start_row_entry = None
end_row_entry = None


def log(message: str):
    log_text.insert(tk.END, message + "\n")
    log_text.see(tk.END)


def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
    if file_path:
        input_path_entry.delete(0, tk.END)
        input_path_entry.insert(0, file_path)


def browser_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        output_path_entry.delete(0, tk.END)
        output_path_entry.insert(0, folder_path)


def show_h5_info():
    file_path = input_path_entry.get()
    if not os.path.exists(file_path):
        messagebox.showerror("错误", "文件路径不存在")
        return

    try:
        log(f"HDF5 info: {file_path}")
        with h5py.File(file_path, "r") as f:
            info_lines = []

            def visitor(name, obj):
                import h5py as _h5py
                if isinstance(obj, _h5py.Group):
                    info_lines.append(f"[GROUP]   {name}")
                elif isinstance(obj, _h5py.Dataset):
                    info_lines.append(f"[DATASET] {name} | shape: {obj.shape}, dtype: {obj.dtype}")

            f.visititems(visitor)

        log("\n".join(info_lines))
    except Exception as e:
        log(f"读取 HDF5 文件失败：{e}")


def show_projection(index: int):
    global projections, image_label

    if projections is None:
        return
    if index < 0 or index >= projections.shape[0]:
        return

    img_arr = projections[index]
    img_norm = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr) + 1e-8) * 255
    img_pil = Image.fromarray(img_norm.astype(np.uint8))

    base_width = 512
    w, h = img_pil.size
    if w > base_width:
        w_percent = base_width / float(w)
        h_size = int(float(h) * w_percent)
        img_pil = img_pil.resize((base_width, h_size), Image.BILINEAR)

    img_tk = ImageTk.PhotoImage(img_pil)
    image_label.config(image=img_tk, width=img_pil.width, height=img_pil.height)
    image_label.image = img_tk
    log(f"显示投影切片 {index}/{projections.shape[0]-1}")


def show_proj():
    global projections, current_slice, slice_slider

    file_path = input_path_entry.get()
    log(f"当前路径设置为：{file_path}")

    if not os.path.exists(file_path):
        log("文件路径无效")
        return

    try:
        with h5py.File(file_path, "r") as f:
            if "/exchange/data" not in f:
                log("HDF5 中不包含 /exchange/data")
                return
            projections = f["/exchange/data"][:]
            current_slice = projections.shape[0] // 2

        log(f"成功读取投影数据: shape={projections.shape}")

        if slice_slider:
            slice_slider.config(to=projections.shape[0] - 1)
            slice_slider.set(current_slice)

        show_projection(current_slice)

    except Exception as e:
        log(f"读取投影图像失败: {e}")


def prev_slice():
    global current_slice
    if projections is not None and current_slice > 0:
        current_slice -= 1
        slice_slider.set(current_slice)
        show_projection(current_slice)


def next_slice():
    global current_slice
    if projections is not None and current_slice < projections.shape[0] - 1:
        current_slice += 1
        slice_slider.set(current_slice)
        show_projection(current_slice)


def slider_changed(val):
    global current_slice
    current_slice = int(float(val))
    show_projection(current_slice)


def show_default_image():
    try:
        default_path = r"E:/Bo/PycharmProjects/utils/Lamino_0000.tif"
        if not os.path.exists(default_path):
            log("Default image not found")
            return

        img_pil = Image.open(default_path)
        img_np = np.array(img_pil)

        if img_np.dtype != np.uint8:
            img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-8) * 255
            img_np = img_np.astype(np.uint8)

        img_pil = Image.fromarray(img_np)

        base_width = 512
        w, h = img_pil.size
        if w > base_width:
            w_percent = base_width / float(w)
            h_size = int(float(h) * w_percent)
            img_pil = img_pil.resize((base_width, h_size), Image.BILINEAR)

        img_tk = ImageTk.PhotoImage(img_pil)
        image_label.config(image=img_tk, width=img_pil.width, height=img_pil.height)
        image_label.image = img_tk
        log("Show default image Lamino_0000.tif")

    except Exception as e:
        log(f"Can not load default image: {e}")


# -------------------------
# TomoCuPy commands (GUI-driven)
# -------------------------
def lamino_try_center():
    input_path = input_path_entry.get()
    output_folder = output_path_entry.get()
    output_path = ru.create_folder(output_folder, "lamino_try_center")

    rotation_axis = rc_entry.get().strip() or "1500"
    center_search_step = css_entry.get().strip() or "1"
    center_search_width = csw_entry.get().strip() or "20"
    tilted_angle = ta_entry.get().strip() or "30"
    binning_number = b_entry.get().strip() or "2"
    start_row = start_row_entry.get().strip() or "1"
    end_row = end_row_entry.get().strip() or "2000"

    print()

    command = [
        "tomocupy", "recon_steps",
        "--file-name", input_path,
        "--out-path-name", output_path,
        "--lamino-angle", tilted_angle,
        "--center-search-width", center_search_width,
        "--center-search-step", center_search_step,
        "--binning", binning_number,
        "--lamino-start-row", start_row,
        "--lamino-end-row", end_row,
        "--reconstruction-type", "try",
        "--nsino-per-chunk", "8",
        "--nproj-per-chunk", "8",
    ]
    subprocess.run(command)


def lamino_try_angle():
    input_path = input_path_entry.get()
    output_folder = output_path_entry.get()
    output_path = ru.create_folder(output_folder, "lamino_angle")

    rotation_axis = rc_entry.get().strip() or "1500"
    angle_search_step = ast_entry.get().strip() or "1"
    angle_search_width = asw_entry.get().strip() or "20"
    tilted_angle = ta_entry.get().strip() or "30"
    binning_number = b_entry.get().strip() or "2"

    command = [
        "tomocupy", "recon_steps",
        "--file-name", input_path,
        "--out-path-name", output_path,
        "--nsino-per-chunk", "8",
        "--nproj-per-chunk", "8",
        "--rotation-axis", rotation_axis,
        "--reconstruction-type", "try_lamino",
        "--lamino-search-width", angle_search_width,
        "--lamino-angle", tilted_angle,
        "--lamino-search-step", angle_search_step,
        "--binning", binning_number,
    ]
    subprocess.run(command)


def lamino_recon():
    input_path = input_path_entry.get()
    output_folder = output_path_entry.get()
    output_path = ru.create_folder(output_folder, "lamino_recon")

    rotation_axis = rc_entry.get().strip() or "1500"
    tilted_angle = ta_entry.get().strip() or "30"
    binning_number = b_entry.get().strip() or "2"

    command = [
        "tomocupy", "recon_steps",
        "--file-name", input_path,
        "--out-path-name", output_path,
        "--rotation-axis", rotation_axis,
        "--lamino-angle", tilted_angle,
        "--binning", binning_number,
        "--reconstruction-type", "full",
        "--nsino-per-chunk", "8",
        "--nproj-per-chunk", "8",
    ]
    subprocess.run(command)

def customized_recon():
    input_path = input_path_entry.get()
    output_folder = output_path_entry.get()
    output_path = ru.create_folder(output_folder, "lamino_recon")

    rotation_axis = rc_entry.get().strip() or "1500"
    tilted_angle = ta_entry.get().strip() or "30"
    binning_number = b_entry.get().strip() or "2"

    command = [
        "tomocupy", "recon_steps",
        "--file-name", input_path,
        "--out-path-name", output_path,
        "--rotation-axis", rotation_axis,
        "--lamino-angle", tilted_angle,
        "--binning", binning_number,
        "--reconstruction-type", "full",
        "--nsino-per-chunk", "8",
        "--nproj-per-chunk", "8",
        ""
    ]
    subprocess.run(command)



def confirm_info():
    input_path = input_path_entry.get()
    output_folder = output_path_entry.get()
    output_path = ru.create_folder(output_folder, "lamino_recon")

    rotation_axis = rc_entry.get().strip() or "1500"
    tilted_angle = ta_entry.get().strip() or "30"
    binning_number = b_entry.get().strip() or "2"
    angle_search_step = ast_entry.get().strip() or "1"
    angle_search_width = asw_entry.get().strip() or "20"

    log("input path:  " + input_path)
    log("output_path: " + output_path)
    log("rotation_axis: " + rotation_axis)
    log(
        "tilted_angle: " + tilted_angle
        + "  angle_search_step:" + angle_search_step
        + "  angle_search_width:" + angle_search_width
    )
    log("binning_number: " + binning_number)


def draw_rounded_dashed_rect(canvas, x1, y1, x2, y2, r=18,
                             dash=(6, 4), outline="black", width=2, tag="decor"):
    r = min(r, (x2-x1)//2, (y2-y1)//2)

    canvas.create_line(x1+r, y1, x2-r, y1, fill=outline, width=width, dash=dash, tags=tag)
    canvas.create_line(x1+r, y2, x2-r, y2, fill=outline, width=width, dash=dash, tags=tag)
    canvas.create_line(x1, y1+r, x1, y2-r, fill=outline, width=width, dash=dash, tags=tag)
    canvas.create_line(x2, y1+r, x2, y2-r, fill=outline, width=width, dash=dash, tags=tag)

    canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90,
                      style="arc", outline=outline, width=width, dash=dash, tags=tag)
    canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90,
                      style="arc", outline=outline, width=width, dash=dash, tags=tag)
    canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90,
                      style="arc", outline=outline, width=width, dash=dash, tags=tag)
    canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90,
                      style="arc", outline=outline, width=width, dash=dash, tags=tag)



def redraw_decor(canvas):
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    if w <= 2 or h <= 2:
        return

    canvas.delete("decor")

    # Scale line width and corner radius with the window size.
    scale = min(w/1200, h/650)
    line_w = max(1, int(2 * scale))
    radius = max(8, int(9 * scale))
    dash = (max(2, int(6 * scale)), max(2, int(4 * scale)))

    for (x1r, y1r, x2r, y2r) in BOXES_REL:
        x1, y1 = int(x1r * w), int(y1r * h)
        x2, y2 = int(x2r * w), int(y2r * h)
        draw_rounded_dashed_rect(
            canvas, x1, y1, x2, y2,
            r=radius, dash=dash, outline="black", width=line_w, tag="decor"
        )
BOXES_REL = [
        (0.01, 0.045, 0.48, 0.20),
        (0.01, 0.25, 0.48, 0.42),
        (0.01, 0.47, 0.48, 0.62),
        (0.01, 0.67, 0.48, 0.7),
        (0.01, 0.75, 0.48, 0.99),
    ]

# ===== Global font settings =====
FONT_SIZE = 6
FONT_FAMILY = "Segoe UI"

FONT_ENTRY = (FONT_FAMILY, FONT_SIZE)
FONT_LABEL = (FONT_FAMILY, FONT_SIZE, "bold")
FONT_BUTTON = (FONT_FAMILY, FONT_SIZE)


def main():
    global input_path_entry, output_path_entry, log_text, image_label
    global rc_entry, css_entry, csw_entry, ta_entry, ast_entry, asw_entry
    global b_entry, start_row_entry, end_row_entry
    global slice_slider

    root = tk.Tk()
    root.title("reconstruct anything @ SSRL 6_2 laminography")

    window_width = int(root.winfo_screenwidth() / 2)
    window_height = int(root.winfo_screenheight() / 2)
    root.geometry(f"{window_width}x{window_height}")
    root.configure(bg="#eaf6fb")

    canvas = tk.Canvas(root, bg="#eaf6fb", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    canvas.bind("<Configure>", lambda e: redraw_decor(canvas))

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Blue.TButton", background="#edf7ff", foreground="#00334d", font=("Segoe UI", 8), padding=5, relief="groove")
    style.configure("Blue.TLabel", background="#eaf6fb", foreground="#00334d", font=("Segoe UI", 8, "bold"), padding=2)
    style.configure("Blue.TEntry", foreground="#00334d", fieldbackground="#ffffff", background="#ffffff", padding=5, font=FONT_LABEL)
    style.configure("Black.TEntry", background="#f0f0f0", foreground="#1c1c1c", fieldbackground="#ffffff", padding=5, font=FONT_LABEL)


    # Shared widget sizes.
    BTN_W, BTN_H = 0.08, 0.035
    ENT_W, ENT_H = 0.08, 0.032
    ENT_W2, ENT_H2 = 0.03, 0.032

    ttk.Label(root, style="Blue.TLabel", text="Micro-CT").place(relx=0.01, rely=0.44)
    ttk.Label(root, style="Blue.TLabel", text="Laminography").place(relx=0.01, rely=0.22)
    ttk.Label(root, style="Blue.TLabel", text="Datasets").place(relx=0.01, rely=0.015)
    ttk.Label(root, style="Blue.TLabel", text="Information").place(relx=0.01, rely=0.72)
    ttk.Label(root, style="Blue.TLabel", text="XANES").place(relx=0.01, rely=0.635)

    # Input file row.
    input_path_entry = ttk.Entry(root, style="Black.TEntry")
    input_path_entry.place(relx=0.02, rely=0.06, relwidth=0.35, relheight=ENT_H)
    ttk.Button(root, text="Input Browser", style="Blue.TButton", command=browse_file)\
        .place(relx=0.38, rely=0.06, relwidth=BTN_W, relheight=BTN_H)

    # Output folder row.
    output_path_entry = ttk.Entry(root, style="Black.TEntry")
    output_path_entry.place(relx=0.02, rely=0.10, relwidth=0.35, relheight=ENT_H)
    ttk.Button(root, text="Output Browser", style="Blue.TButton", command=browser_folder)\
        .place(relx=0.38, rely=0.10, relwidth=BTN_W, relheight=BTN_H)

    # Dataset control buttons.
    line2 = 0.15
    ttk.Button(root, text="Show H5 Info", style="Blue.TButton", command=show_h5_info)\
        .place(relx=0.02, rely=line2, relwidth=BTN_W, relheight=BTN_H)
    ttk.Button(root, text="Show Data", style="Blue.TButton", command=show_proj)\
        .place(relx=0.12, rely=line2, relwidth=BTN_W, relheight=BTN_H)
    ttk.Button(root, text="Confirm Info", style="Blue.TButton", command=confirm_info)\
        .place(relx=0.22, rely=line2, relwidth=BTN_W, relheight=BTN_H)
    ttk.Button(root, text="Tif to H5", style="Blue.TButton", command=ru.tif_2_h5_clicked) \
        .place(relx=0.32, rely=line2, relwidth=BTN_W, relheight=BTN_H)

    # Rotation-center search row.
    line3 = 0.265
    ttk.Label(root, style="Blue.TLabel", text="Rotation Center").place(relx=0.02, rely=line3)
    rc_entry = ttk.Entry(root, style="Blue.TEntry")
    rc_entry.insert(0, "")
    rc_entry.place(relx=0.11, rely=line3, relwidth=ENT_W2, relheight=ENT_H)

    ttk.Label(root, style="Blue.TLabel", text="Step").place(relx=0.18, rely=line3)
    css_entry = ttk.Entry(root, style="Blue.TEntry")
    css_entry.insert(0, "")
    css_entry.place(relx=0.22, rely=line3, relwidth=ENT_W2, relheight=ENT_H)

    ttk.Label(root, style="Blue.TLabel", text="Range").place(relx=0.28, rely=line3)
    csw_entry = ttk.Entry(root, style="Blue.TEntry")
    csw_entry.insert(0, "")
    csw_entry.place(relx=0.325, rely=line3, relwidth=ENT_W2, relheight=ENT_H)

    ttk.Button(root, text="Search", style="Blue.TButton", command=lamino_try_center) \
        .place(relx=0.375, rely=line3, relwidth=BTN_W, relheight=BTN_H)



    # Lamino-angle search row.
    line4 = 0.315
    ttk.Label(root, style="Blue.TLabel", text="Tilted Angle").place(relx=0.02, rely=line4)
    ta_entry = ttk.Entry(root, style="Blue.TEntry")
    ta_entry.insert(0, "")
    ta_entry.place(relx=0.11, rely=line4, relwidth=ENT_W2, relheight=ENT_H)

    ttk.Label(root, style="Blue.TLabel", text="Step").place(relx=0.18, rely=line4)
    ast_entry = ttk.Entry(root, style="Blue.TEntry")
    ast_entry.insert(0, "")
    ast_entry.place(relx=0.22, rely=line4, relwidth=ENT_W2, relheight=ENT_H)

    ttk.Label(root, style="Blue.TLabel", text="Range").place(relx=0.28, rely=line4)
    asw_entry = ttk.Entry(root, style="Blue.TEntry")
    asw_entry.insert(0, "")
    asw_entry.place(relx=0.325, rely=line4, relwidth=ENT_W2, relheight=ENT_H)

    ttk.Button(root, text="Search", style="Blue.TButton", command=lamino_try_angle)\
        .place(relx=0.375, rely=line4, relwidth=BTN_W, relheight=BTN_H)

    # Reconstruction parameter row.
    line5 = 0.365
    ttk.Label(root, style="Blue.TLabel", text="Binning").place(relx=0.02, rely=line5)
    b_entry = ttk.Entry(root, style="Blue.TEntry")
    b_entry.insert(0, "")
    b_entry.place(relx=0.11, rely=line5, relwidth=ENT_W2, relheight=ENT_H)

    ttk.Label(root, style="Blue.TLabel", text="Start row").place(relx=0.16, rely=line5)
    start_row_entry = ttk.Entry(root, style="Blue.TEntry")
    start_row_entry.insert(0, "")
    start_row_entry.place(relx=0.22, rely=line5, relwidth=ENT_W2, relheight=ENT_H)

    ttk.Label(root, style="Blue.TLabel", text="End row").place(relx=0.27, rely=line5)
    end_row_entry = ttk.Entry(root, style="Blue.TEntry")
    end_row_entry.insert(0, "")
    end_row_entry.place(relx=0.325, rely=line5, relwidth=ENT_W2, relheight=ENT_H)

    ttk.Button(root, text="Reconstruction", style="Blue.TButton", command=lamino_recon) \
        .place(relx=0.375, rely=line5, relwidth=BTN_W, relheight=BTN_H)

    # Micro-CT algorithm row.
    # -------------------------
    # Micro-CT algorithm selector.
    # -------------------------
    line6= 0.5-0.02
    microct_algo = tk.StringVar(value="FBP")



    style.configure(
        "Blue.TRadiobutton",
        background="#eaf6fb",
        foreground="#00334d",
        font=("Segoe UI", 8)
    )

    # Algorithm choices.

    ttk.Label(root, style="Blue.TLabel", text="Algorithm").place(relx=0.02, rely=line6)
    ttk.Radiobutton(root, style= "Blue.TRadiobutton", text="FBP",  variable=microct_algo, value="FBP").place(relx=0.02+0.1, rely=line6)
    ttk.Radiobutton(root, style= "Blue.TRadiobutton", text="SIRT", variable=microct_algo, value="SIRT").place(relx=0.12+0.1, rely=line6)
    ttk.Radiobutton(root, style= "Blue.TRadiobutton", text="SART", variable=microct_algo, value="SART").place(relx=0.22+0.1, rely=line6)

    # Optional iteration settings.
    # ttk.Label(root, style="Blue.TLabel", text="Iterations").place(relx=0.54, rely=0.78)
    # microct_iter_entry = ttk.Entry(root, style="Blue.TEntry")
    # microct_iter_entry.insert(0, "50")
    # microct_iter_entry.place(relx=0.02, rely=0.78, relwidth=0.06, relheight=0.035)

    # Placeholder for the run button.




    # Micro-CT rotation-center row.
    line3 = 0.575-0.03-0.02
    ttk.Label(root, style="Blue.TLabel", text="Rotation Center").place(relx=0.02, rely=line3)
    rc_entry = ttk.Entry(root, style="Blue.TEntry")
    rc_entry.insert(0, "")
    rc_entry.place(relx=0.11, rely=line3, relwidth=ENT_W2, relheight=ENT_H)

    ttk.Label(root, style="Blue.TLabel", text="Step").place(relx=0.18, rely=line3)
    css_entry = ttk.Entry(root, style="Blue.TEntry")
    css_entry.insert(0, "")
    css_entry.place(relx=0.22, rely=line3, relwidth=ENT_W2, relheight=ENT_H)

    ttk.Label(root, style="Blue.TLabel", text="Range").place(relx=0.28, rely=line3)
    csw_entry = ttk.Entry(root, style="Blue.TEntry")
    csw_entry.insert(0, "")
    csw_entry.place(relx=0.325, rely=line3, relwidth=ENT_W2, relheight=ENT_H)

    ttk.Button(root, text="Search", style="Blue.TButton", command=lamino_try_center) \
        .place(relx=0.375, rely=line3, relwidth=BTN_W, relheight=BTN_H)

    # Micro-CT reconstruction row.
    line5= 0.625-0.03-0.02
    ttk.Label(root, style="Blue.TLabel", text="Binning").place(relx=0.02, rely=line5)
    b_entry = ttk.Entry(root, style="Blue.TEntry")
    b_entry.insert(0, "")
    b_entry.place(relx=0.11, rely=line5, relwidth=ENT_W2, relheight=ENT_H)

    ttk.Label(root, style="Blue.TLabel", text="Start row").place(relx=0.16, rely=line5)
    start_row_entry = ttk.Entry(root, style="Blue.TEntry")
    start_row_entry.insert(0, "")
    start_row_entry.place(relx=0.22, rely=line5, relwidth=ENT_W2, relheight=ENT_H)

    ttk.Label(root, style="Blue.TLabel", text="End row").place(relx=0.27, rely=line5)
    end_row_entry = ttk.Entry(root, style="Blue.TEntry")
    end_row_entry.insert(0, "")
    end_row_entry.place(relx=0.325, rely=line5, relwidth=ENT_W2, relheight=ENT_H)

    ttk.Button(root, text="Reconstruction", style="Blue.TButton", command=lamino_recon) \
        .place(relx=0.375, rely=line5, relwidth=BTN_W, relheight=BTN_H)


    # Log area.
    log_text = scrolledtext.ScrolledText(root)
    log_text.place(relx=0.01, rely=0.75, relheight=0.238, relwidth=0.47)

    # Image preview area.
    image_label = tk.Label(root, text="no load image", bg="gray")
    image_label.place(relx=0.5, rely=0.02, relwidth=0.45, relheight=0.8)

    ttk.Button(root, text="prev_image", style="Blue.TButton", command=prev_slice)\
        .place(relx=0.85, rely=0.84, relwidth=BTN_W, relheight=BTN_H)
    ttk.Button(root, text="next_image", style="Blue.TButton", command=next_slice)\
        .place(relx=0.85, rely=0.9, relwidth=BTN_W, relheight=BTN_H)

    # Projection slider.
    slice_slider = tk.Scale(root, from_=0, to=0, orient="horizontal", label="browser_bar",
                            command=slider_changed, length=300)
    slice_slider.place(relx=0.5, rely=0.83)

    show_default_image()
    root.mainloop()

if __name__ == "__main__":
    main()