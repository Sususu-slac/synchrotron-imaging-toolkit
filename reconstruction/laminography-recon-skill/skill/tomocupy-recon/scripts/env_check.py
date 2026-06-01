#!/usr/bin/env python3
"""Environment check for TomoCuPy recon skill.

This script is meant to be run by humans (or CI) to validate that a machine can
run `tomocupy recon` via conda.

It checks:
- conda executable exists
- conda env exists
- `tomocupy --help` works inside the env

Usage:
  python env_check.py --conda /path/to/conda --env tomocupy_ubuntu_cuda118

If --conda is omitted, it uses $TOMOCUPY_CONDA or falls back to `conda` in PATH.
If --env is omitted, it uses $TOMOCUPY_ENV or defaults to tomocupy_ubuntu_cuda118.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conda", default=os.environ.get("TOMOCUPY_CONDA", "conda"))
    ap.add_argument("--env", default=os.environ.get("TOMOCUPY_ENV", "tomocupy_ubuntu_cuda118"))
    args = ap.parse_args()

    conda = args.conda
    env = args.env

    # resolve conda
    if conda != "conda":
        if not os.path.exists(conda):
            print(f"ERROR: conda not found at: {conda}")
            return 2
    else:
        if shutil.which("conda") is None:
            print("ERROR: conda not found in PATH. Provide --conda or set TOMOCUPY_CONDA")
            return 2

    print(f"Conda: {conda}")
    print(f"Env  : {env}")

    # list envs
    code, out = run([conda, "env", "list"])
    if code != 0:
        print("ERROR: failed to run `conda env list`\n" + out)
        return 3

    if env not in out:
        print("ERROR: env not found in conda env list")
        print(out)
        return 4

    # tomocupy help
    code, out = run([conda, "run", "--no-capture-output", "-n", env, "tomocupy", "--help"])
    if code != 0:
        print("ERROR: `tomocupy --help` failed inside env")
        print(out)
        return 5

    print("OK: tomocupy is runnable in this environment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
