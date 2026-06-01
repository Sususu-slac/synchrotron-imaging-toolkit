#!/usr/bin/env python3
"""Build TomoCuPy recon commands with consistent defaults and outdir naming.

This is used by the skill logic (or manually) to generate the exact command line
and output folder.

It does NOT execute the command.

Examples:
  python build_cmd.py try  /data/X.hdf5
  python build_cmd.py full /data/X.hdf5 --center 1032.5

Environment:
  TOMOCUPY_CONDA, TOMOCUPY_ENV
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path


def fmt_center_tag(center: float) -> str:
    # 1032.5 -> 1032p5
    s = f"{center:.10g}"
    return s.replace(".", "p")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["try", "full"])
    ap.add_argument("file", help="Path to .h5/.hdf5")

    ap.add_argument("--conda", default=os.environ.get("TOMOCUPY_CONDA", "/home/subo/miniforge3/bin/conda"))
    ap.add_argument("--env", default=os.environ.get("TOMOCUPY_ENV", "tomocupy_ubuntu_cuda118"))

    ap.add_argument("--nsino", type=int, default=16)
    ap.add_argument("--stripe", default="vo-all")
    ap.add_argument("--stamp", default=None)

    # try
    ap.add_argument("--center-search-width", type=float, default=300)
    ap.add_argument("--center-search-step", type=float, default=0.5)

    # full
    ap.add_argument("--center", type=float, default=None)

    args = ap.parse_args()

    file = Path(args.file)
    if args.stamp:
        stamp = args.stamp
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")

    # outdir base: sibling folder with stem name
    base = file.parent / file.stem

    if args.mode == "try":
        outdir = base / f"recon_try_cuda118_nsino{args.nsino}_voall_cw{int(args.center_search_width)}_cs{str(args.center_search_step).replace('.', 'p')}_{stamp}"
        cmd = (
            f"{args.conda} run -n {args.env} tomocupy recon "
            f"--file-name \"{file}\" "
            f"--out-path-name \"{outdir}\" "
            f"--reconstruction-type try "
            f"--nsino-per-chunk {args.nsino} "
            f"--remove-stripe-method {args.stripe} "
            f"--center-search-width {args.center_search_width} "
            f"--center-search-step {args.center_search_step} "
            f"--verbose "
            f"> \"{outdir}/run.log\" 2>&1"
        )
    else:
        if args.center is None:
            raise SystemExit("--center is required for full")
        ctag = fmt_center_tag(args.center)
        outdir = base / f"recon_full_cuda118_center{ctag}_nsino{args.nsino}_voall_{stamp}"
        cmd = (
            f"{args.conda} run -n {args.env} tomocupy recon "
            f"--file-name \"{file}\" "
            f"--out-path-name \"{outdir}\" "
            f"--reconstruction-type full "
            f"--rotation-axis {args.center} "
            f"--nsino-per-chunk {args.nsino} "
            f"--remove-stripe-method {args.stripe} "
            f"--verbose "
            f"> \"{outdir}/run.log\" 2>&1"
        )

    print("OUTDIR:")
    print(str(outdir))
    print("\nCMD:")
    print(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
