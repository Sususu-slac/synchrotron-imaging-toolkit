---
name: tomocupy-recon
description: Build and run TomoCuPy tomography reconstruction commands (recon try / recon full) with standard defaults, confirmation prompts, logging, and consistent output folder naming. Use when the user says "recon try <H5>" or "recon full <H5>" and wants you to assemble the exact tomocupy command, ask for confirmation, then execute and track progress.
---

# TomoCuPy recon (try/full) workflow

This skill standardizes running `tomocupy recon` for HDF5 tomography datasets.

## Inputs
- User says: `recon try <h5_path>` or `recon full <h5_path>`.
- **Try**: run center search around the (optional) provided center using width/step.
- **Full**: requires a rotation center; if user did not provide one, ask.

## Defaults (can be overridden)
- Conda executable: `${TOMOCUPY_CONDA}` (fallback: `conda` in PATH)
- Conda env: `${TOMOCUPY_ENV}` (fallback: `tomocupy_ubuntu_cuda118`)
- Common flags:
  - `--remove-stripe-method vo-all`
  - `--nsino-per-chunk 16`
  - `--verbose`
- Try-mode center search defaults:
  - `--center-search-width 300`
  - `--center-search-step 0.5`

## Output path convention
Given input file `.../X.hdf5`:
- outdir base: `dirname(X)/X/`
- `try` outdir: `recon_try_cuda118_nsino16_voall_cw<WIDTH>_cs<STEP>_<STAMP>`
- `full` outdir: `recon_full_cuda118_center<CENTER>_nsino16_voall_<STAMP>`

`<STAMP>` uses `YYYYMMDD_HHMM`.

## Required confirmation step
Before execution, always print:
1) exact command (single line)
2) output folder
3) key parameters (try: width/step; full: center)
Then ask: **Run it? (yes/no)**

## Execution and logs
- Run via `conda run -n <env> tomocupy recon ...`
- Always write logs to `<OUTDIR>/run.log`.

## Helper scripts
- `scripts/env_check.py` – validate conda env + tomocupy availability.
- `scripts/build_cmd.py` – generate command + outdir naming consistently.
