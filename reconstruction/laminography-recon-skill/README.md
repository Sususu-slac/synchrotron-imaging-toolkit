# laminography skill (repo bundle)

This repository bundle is intended to be uploaded to GitHub as-is.

It contains:

- `skill/tomocupy-recon/` – the Clawdbot skill package (SKILL.md + scripts)
- `tools/` – optional helper notes and templates

## What this skill does

Adds a standardized workflow for TomoCuPy reconstructions:

- `recon try <file.h5>`
- `recon full <file.h5> center=<rotation_axis>`

The skill:

1) builds the exact `tomocupy recon` command line
2) prints the command + output folder + key params
3) asks for confirmation
4) runs the command and logs to `OUTDIR/run.log`

## What users must install

Each user/machine must install and validate TomoCuPy themselves (GPU/CUDA, conda env, `tomocupy` CLI). The skill only orchestrates command building + execution.

## Environment variables (recommended)

To avoid hard-coding machine-specific paths, set:

- `TOMOCUPY_CONDA` – path to the conda executable (or just `conda` if in PATH)
- `TOMOCUPY_ENV` – conda env name containing TomoCuPy

Example:

```bash
export TOMOCUPY_CONDA="$HOME/miniforge3/bin/conda"
export TOMOCUPY_ENV="tomocupy_ubuntu_cuda118"
```

## Preflight environment check

Run:

```bash
python skill/tomocupy-recon/scripts/env_check.py \
  --conda "$TOMOCUPY_CONDA" \
  --env "$TOMOCUPY_ENV"
```

It will verify:
- conda works
- env exists
- `tomocupy --help` works inside the env

## Generate a command (no execution)

```bash
python skill/tomocupy-recon/scripts/build_cmd.py try  /path/to/X.hdf5
python skill/tomocupy-recon/scripts/build_cmd.py full /path/to/X.hdf5 --center 1032.5
```

## Installing into Open Clawdbot (copy folder method)

You chose the **copy-to-skills-directory** install method.

### What to copy

Copy this folder:

- `skill/tomocupy-recon/`

into your Open Clawdbot skills directory so that it becomes:

- `<SKILLS_DIR>/tomocupy-recon/SKILL.md`

### Where is `<SKILLS_DIR>`?

It depends on your Open Clawdbot setup. Common patterns are:

- a repo-local folder (e.g. `./skills/`)
- a user config folder (e.g. `~/.clawdbot/skills/`)

If you are not sure, search for existing skills on that machine and place `tomocupy-recon/` next to them.

### Quick self-test

After copying, trigger the skill by asking the agent:

- `recon try /path/to/file.h5`

and confirm it prints the command + asks for confirmation.
