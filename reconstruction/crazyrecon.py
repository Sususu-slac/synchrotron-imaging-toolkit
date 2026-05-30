import os
import h5py
import numpy as np
from glob import glob
# from skimage.io import imread
from tkinter import filedialog, Tk, messagebox, ttk
from datetime import datetime
import tkinter
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import h5py
import numpy as np
import os
import subprocess
from datetime import datetime
from tkinter import ttk

def create_folder(path, folder_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{folder_name}_{timestamp}"
    folder_path = os.path.join(path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def lamino_recon(input_path, output_path, ra, ta, b):

    input_path = input_path
    output_folder = output_path
    foldername = "lamino_recon_"+ta
    output_path = create_folder(output_folder, foldername)
    rotation_axis = ra
    tilted_angle = ta
    binning_number = b

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
        "--remove-stripe-method", "vo-all",
        ]
    subprocess.run(command)
    return 0

def lamino_try_angle(input_path,output_path,rotation_axis,angle_search_width,tilted_angle,angle_search_step):

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
        "--remove-stripe-method", "vo-all",
        "--end-row","185",
        "--start-row","1",
    ]
    subprocess.run(command)
    return 0

def GoCracy(inputpath, outpath, ra_start, ra_end, ra_step, bining, ta):
    input_path = inputpath
    output_path = outpath
    ra_start = str(ra_start)
    ra_end = str(ra_end)
    ra_step = str(ra_step)
    b = str(bining)
    ta = str(ta)
    for i in range(ra_start, ra_end, ra_step):
        ra = str(i)
        lamino_recon(input_path, output_path, ra, ta, b)
        print('cycle'+ra)



#
# input_path = r'D:/Datasets/RedP_Reconstructed_Results/h5/BM_BlackP_pre.h5'
# output_path = r'D:/Datasets/RedP_Reconstructed_Results/Results3/BM_BlackP_pre'
#
# ta = '-30'
# b = '1'
# for i in range(900, 1300, 2):
#     ra = str(i)
#     lamino_recon(input_path, output_path, ra, ta, b)
#
#
# input_path = r'D:\Datasets\RedP_Reconstructed_Results\h5\BM_BlackP_pristine.h5'
# output_path = r'D:/Datasets/RedP_Reconstructed_Results/Results3/BM_BlackP_pristine'
#
# ta = '-30'
# b = '1'
# for i in range(900, 1300, 2):
#     ra = str(i)
#     lamino_recon(input_path, output_path, ra, ta, b)
#
#
# input_path = r'D:/Datasets/RedP_Reconstructed_Results/h5/BM_BlackP_Charged.h5'
# output_path = r'D:/Datasets/RedP_Reconstructed_Results/Results3/BM_BlackP_Charged'
#
# ta = '-30'
# b = '1'
# for i in range(900, 1300,2):
#     ra = str(i)
#     lamino_recon(input_path, output_path, ra, ta, b)

# input_path = r'D:\Datasets\RedP_Reconstructed_Results\h5\Bulk_Charged.h5'
# output_path = r'D:/Datasets/RedP_Reconstructed_Results/Results4/Bulk_Charged'
#
#
# rc = "942"
# b = '1'
# for i in np.arange(-36, -30, 0.5):
#     ta = str(i)
#     lamino_recon(input_path, output_path, rc, ta, b)
# lamino_try_angle(input_path, output_path, rc, "50", "-30", '1')

input_path = r'D:\Datasets\RedP_Reconstructed_Results\h5\Bulk_Discharged.h5'
output_path = r'D:/Datasets/RedP_Reconstructed_Results/Results4/Bulk_Discharged'
# ta = '-30'
rc = "932"
# ta = '-30'
b = '2'
for i in np.arange(-30, -28, 0.1):
    ta = str(i)
    lamino_recon(input_path, output_path, rc, ta, b)
# lamino_try_angle(input_path, output_path, rc, "50", "-30", '1')
#
# input_path = r'D:/Datasets/RedP_Reconstructed_Results/h5/Bulk_pre.h5'
# output_path = r'D:/Datasets/RedP_Reconstructed_Results/Results4/Bulk_pre'
# # ta = '-30'
# rc = "902"
# # ta = '-30'
# b = '1'
# for i in range(-36, -15, 1):
#     ta = str(i)
#     lamino_recon(input_path, output_path, rc, ta, b)
# # lamino_try_angle(input_path, output_path, rc, "50", "-30", '1')
#
# input_path = r'D:/Datasets/RedP_Reconstructed_Results/h5/Bulk_Pristine.h5'
# output_path = r'D:/Datasets/RedP_Reconstructed_Results/Results4/Bulk_Pristine'
# # ta = '-30'
# rc = "1050"
# # ta = '-30'
# b = '1'
# for i in range(-36, -15, 1):
#     ta = str(i)
#     print(ta)
#     lamino_recon(input_path, output_path, rc, ta, b)
# lamino_try_angle(input_path, output_path, rc, "50", "-30", '1')

