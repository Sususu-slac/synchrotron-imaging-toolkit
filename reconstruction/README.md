# Laminography Reconstruction Toolkit

A GUI toolkit for synchrotron tomography and laminography reconstruction using TomoCuPy.
Tomocupy website https://tomocupy.readthedocs.io/en/latest/

This toolkit helps users load HDF5 datasets, preview projection images, search reconstruction parameters, and run TomoCuPy reconstruction commands through a user-friendly GUI.

---

## Files

```text
reconstruction/
├── laminography_recon.py     # Main GUI application
├── recon_utilities.py        # Utility functions for H5 conversion and analysis
├── crazyrecon.py             # Experimental batch reconstruction script
└── README.md
```

Recommended usage:

- Use `laminography_recon.py` for normal reconstruction work.
- Use `recon_utilities.py` as a utility toolbox.
- Use `crazyrecon.py` only for testing or experimental batch reconstruction.

---

## Requirements

Make sure **TomoCuPy** is installed and available in your terminal:

```bash
tomocupy --help
```

Install the required Python packages:

```bash
pip install numpy h5py pillow tifffile matplotlib opencv-python scikit-image
```

If you use conda, activate the environment where TomoCuPy is installed:

```bash
conda activate tomocupy
```

> The GUI should be launched from the same environment where `tomocupy` works.

---

## Expected HDF5 Format

The input `.h5` file should include:

```text
/exchange/data        # Projection images, shape: (nproj, height, width)
/exchange/data_white  # Flat-field images
/exchange/data_dark   # Dark-field images
/exchange/theta       # Projection angles
```

---

## Run the GUI

```bash
python laminography_recon.py
```

---

## Basic Workflow

1. Select an input `.h5` file.
2. Select an output folder.
3. Click **Show H5 Info** to check the file structure.
4. Click **Show Data** to preview projection images.
5. Search the rotation center.
6. Search the laminography angle.
7. Run full reconstruction.

---

## Main Functions

### 1. Rotation Center Search

Searches for a suitable rotation center using TomoCuPy:

```bash
tomocupy recon_steps --reconstruction-type try
```

Main parameters:

- Rotation center
- Center search width
- Center search step
- Laminography angle
- Binning
- Start row / end row

---

### 2. Laminography Angle Search

Searches for a suitable laminography angle:

```bash
tomocupy recon_steps --reconstruction-type try_lamino
```

Main parameters:

- Rotation axis
- Laminography angle
- Angle search width
- Angle search step
- Binning

---

### 3. Full Reconstruction

Runs the final reconstruction:

```bash
tomocupy recon_steps --reconstruction-type full
```

The reconstruction results will be saved in the selected output folder.

---

## Utilities

`recon_utilities.py` provides helper functions for:

- Converting TIFF image folders to HDF5
- Generating `theta.txt`
- Checking HDF5 data
- Scoring reconstruction results
- Finding a good rotation center

---

## Notes

- `crazyrecon.py` is experimental. Use it only if you understand the code.
- If you see `tomocupy: not found`, activate the correct conda environment first.
- Large datasets may require enough GPU memory.
- The HDF5 file structure should be checked before reconstruction.

---

## License

MIT License
