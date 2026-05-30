# recon_utilities.py
import os
import re
import glob
from glob import glob as glob2
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import shannon_entropy
from tkinter import filedialog, Tk, messagebox


# -------------------------
# Folder helpers
# -------------------------
def create_folder(path: str, folder_name: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{folder_name}_{timestamp}"
    folder_path = os.path.join(path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


# -------------------------
# H5 packing (proj/flat/dark + theta)
# -------------------------
def load_images(folder: str, ext: str = "tif") -> np.ndarray:
    paths = sorted(glob2(os.path.join(folder, f"*.{ext}")))
    if not paths:
        raise ValueError(f"找不到图像文件于: {folder}")
    return np.array([imread(p) for p in paths])


def pack_to_h5(
    proj_dir: str,
    flat_dir: str,
    dark_dir: str,
    theta_path: str,
    output_path: str,
    dtype: str = "uint16",
) -> None:
    proj = load_images(proj_dir).astype(dtype)
    flat = load_images(flat_dir).astype(dtype)
    dark = load_images(dark_dir).astype(dtype)

    theta = np.load(theta_path) if theta_path.endswith(".npy") else np.loadtxt(theta_path)
    if proj.shape[0] != theta.shape[0]:
        raise ValueError(f"投影数 {proj.shape[0]} 与角度数 {theta.shape[0]} 不一致")

    with h5py.File(output_path, "w") as f:
        exch = f.create_group("/exchange")
        exch.create_dataset("data", data=proj, compression="gzip")
        exch.create_dataset("data_white", data=flat, compression="gzip")
        exch.create_dataset("data_dark", data=dark, compression="gzip")
        exch.create_dataset("theta", data=theta.astype(np.float32))


def tif2h5_dialog() -> None:
    """交互式：选择 proj/flat/dark/theta，保存成 h5"""
    root = Tk()
    root.withdraw()

    try:
        proj_dir = filedialog.askdirectory(title="选择 projection 图像文件夹")
        if not proj_dir:
            return

        flat_dir = filedialog.askdirectory(title="选择 flat 图像文件夹")
        if not flat_dir:
            return

        dark_dir = filedialog.askdirectory(title="选择 dark 图像文件夹")
        if not dark_dir:
            return

        theta_path = filedialog.askopenfilename(
            title="选择 theta.txt 或 .npy 文件",
            filetypes=[("Text files", "*.txt"), ("NumPy files", "*.npy")],
        )
        if not theta_path:
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5")],
            title="保存为 HDF5 文件",
        )
        if not save_path:
            return

        pack_to_h5(proj_dir, flat_dir, dark_dir, theta_path, save_path)
        messagebox.showinfo("完成", f"成功打包 HDF5 文件：\n{save_path}")

    except Exception as e:
        messagebox.showerror("错误", f"打包失败:\n{e}")


# -------------------------
# theta generation
# -------------------------
def generate_theta() -> str | None:
    """选择投影文件夹 -> 自动生成 theta.txt（0~360）"""
    root = Tk()
    root.withdraw()

    proj_dir = filedialog.askdirectory(title="Select projection folder")
    if not proj_dir:
        return None
    folder = Path(proj_dir)

    save_dir = filedialog.askdirectory(title="Select folder to save theta.txt")
    if not save_dir:
        return None
    save_dir = Path(save_dir)

    proj_num = sum(
        1 for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in [".tif", ".tiff"]
    )
    if proj_num < 2:
        messagebox.showerror("Error", "Not enough projections to generate theta.")
        return None

    step = 360.0 / (proj_num - 1)
    theta = [i * step for i in range(proj_num)]

    theta_path = save_dir / "theta.txt"
    with open(theta_path, "w") as f:
        for a in theta:
            f.write(f"{a:.6f}\n")

    messagebox.showinfo("Done", f"theta.txt saved:\n{theta_path}")
    return str(theta_path)


# -------------------------
# Rotation center scoring
# -------------------------
def load_slice_set_to_numpy(folder: str, pattern: str) -> tuple[np.ndarray, np.ndarray]:
    """读取 recon_slice###_center###.tif(f) 并按 center 排序"""
    file_list = sorted(glob.glob(f"{folder}/{pattern}"))
    if not file_list:
        raise FileNotFoundError("未找到匹配图像，请检查路径或命名模式。")

    imgs, centers = [], []
    for f in file_list:
        m = re.search(r"slice(\d+)_center(\d+\.?\d*)", f)
        if not m:
            continue
        centers.append(float(m.group(2)))
        imgs.append(tifffile.imread(f).astype(np.float32))

    if not imgs:
        raise ValueError("无法加载任何有效图像。")

    idx = np.argsort(centers)
    imgs_np = np.stack([imgs[i] for i in idx], axis=0)
    centers = np.array(centers)[idx]
    return imgs_np, centers


def symmetry_score(img: np.ndarray) -> float:
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_flip = np.fliplr(img)
    return float(np.corrcoef(img.ravel(), img_flip.ravel())[0, 1])


def laplacian_var(img: np.ndarray) -> float:
    img = img.astype(np.float64)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def entropy_score(img: np.ndarray) -> float:
    return float(shannon_entropy(img))


def normalize(arr: np.ndarray) -> np.ndarray:
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-12)


def find_rc(
    pattern: str = "recon_slice*_center*.tif*",
    w_sym: float = 0.6,
    w_lap: float = 0.3,
    w_ent: float = 0.1,
) -> float | None:
    """
    选择 try-center 输出文件夹，自动评分并画图，返回 best_center
    """
    root = Tk()
    root.withdraw()

    folder = filedialog.askdirectory(title="Select folder containing recon_slice*_center*.tif")
    if not folder:
        return None

    try:
        imgs, centers = load_slice_set_to_numpy(folder, pattern)

        sym_scores, lap_scores, ent_scores = [], [], []
        for img in imgs:
            sym_scores.append(symmetry_score(img))
            lap_scores.append(laplacian_var(img))
            ent_scores.append(entropy_score(img))

        sym_scores = np.array(sym_scores)
        lap_scores = np.array(lap_scores)
        ent_scores = np.array(ent_scores)

        sym_n = normalize(sym_scores)
        lap_n = normalize(lap_scores)
        ent_n = normalize(ent_scores)

        combined = w_sym * sym_n + w_lap * lap_n + w_ent * ent_n
        best_idx = int(np.argmax(combined))
        best_center = float(centers[best_idx])

        # plot
        plt.figure(figsize=(8, 4))
        plt.plot(centers, sym_n, "-o", label="Symmetry (norm)")
        plt.plot(centers, lap_n, "--o", label="Laplacian Var (norm)")
        plt.plot(centers, ent_n, ":o", label="Entropy (norm)")
        plt.plot(centers, combined, "-o", linewidth=2, color="black", label="Combined Score")
        plt.axvline(best_center, color="r", linestyle="--", label=f"Best center = {best_center:.2f}")
        plt.xlabel("Rotation center")
        plt.ylabel("Normalized Score")
        plt.title("Automatic rotation center scoring")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        messagebox.showinfo("Result", f"Best rotation center = {best_center:.2f}")
        return best_center

    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None


# -------------------------
# dataset_dir/proj flat dark theta -> h5
# -------------------------
def _load_tif_folder(folder: str) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.tiff")))
    if not files:
        raise RuntimeError(f"No tif or tiff found in {folder}")
    imgs = [imread(f) for f in files]
    arr = np.stack(imgs, axis=0) if len(imgs) > 1 else imgs[0]
    return arr.astype(np.float32)


def _ensure_3d(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    return arr[None, ...] if arr.ndim == 2 else arr


def tif_2_h5(dataset_dir: str) -> str:
    dataset_dir = os.path.abspath(dataset_dir)
    proj_dir = os.path.join(dataset_dir, "proj")
    flat_dir = os.path.join(dataset_dir, "flat")
    dark_dir = os.path.join(dataset_dir, "dark")
    theta_file = os.path.join(dataset_dir, "theta", "theta.txt")
    h5_dir = os.path.join(dataset_dir, "h5")

    if not os.path.isdir(proj_dir):
        raise RuntimeError("Missing proj/ folder")
    if not os.path.isfile(theta_file):
        raise RuntimeError("Missing theta/theta.txt")

    os.makedirs(h5_dir, exist_ok=True)

    data = _ensure_3d(_load_tif_folder(proj_dir))
    n_proj, H, W = data.shape

    theta = np.loadtxt(theta_file).astype(np.float32)
    if len(theta) != n_proj:
        raise ValueError("theta length does not match projections")

    flat = _ensure_3d(_load_tif_folder(flat_dir)) if os.path.isdir(flat_dir) else None
    dark = _ensure_3d(_load_tif_folder(dark_dir)) if os.path.isdir(dark_dir) else None

    out_name = os.path.basename(dataset_dir.rstrip("/\\")) + ".h5"
    out_path = os.path.join(h5_dir, out_name)

    with h5py.File(out_path, "w") as f:
        exch = f.create_group("exchange")
        exch.create_dataset("data", data=data, chunks=(1, H, W), compression="gzip", compression_opts=4)
        exch.create_dataset("theta", data=theta)

        if dark is not None:
            exch.create_dataset("dark", data=dark, chunks=(1, H, W), compression="gzip", compression_opts=4)
        if flat is not None:
            exch.create_dataset("flat", data=flat, chunks=(1, H, W), compression="gzip", compression_opts=4)

        f.attrs["source"] = "tif_2_h5"
        f.attrs["dtype"] = "float32"

    return out_path


def tif_2_h5_clicked() -> None:
    root = Tk()
    root.withdraw()

    dataset_dir = filedialog.askdirectory(title="Select dataset folder")
    if not dataset_dir:
        return

    try:
        out_h5 = tif_2_h5(dataset_dir)
        messagebox.showinfo("Done", f"Saved:\n{out_h5}")
    except Exception as e:
        messagebox.showerror("Failed", str(e))