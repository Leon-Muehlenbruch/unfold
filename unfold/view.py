#!/usr/bin/env python3
"""
unf-view — view WaterGAP UNF files (map, stats, PNG/CSV export).

Examples:
  unf-view unf/G_SURFACE_RUNOFF/G_SURFACE_RUNOFF_2011.12.UNF0            # map (annual mean)
  unf-view unf/G_SURFACE_RUNOFF/G_SURFACE_RUNOFF_2011.12.UNF0 --layer 7  # July only
  unf-view unf/OTHER_UNF_FILES/G_PWEATHERING.UNF0 --stats                # stats only
  unf-view file.UNF0 --save map.png                                      # PNG instead of window
  unf-view file.UNF0 --log                                               # log colour scale

The data type is derived from the extension:
  .UNF0 = float32 big-endian   .UNF1 = int8   .UNF2 = int16 BE   .UNF4 = int32 BE
Monthly files (".12." in the name, or size = 12×cell count) are detected;
without --layer the 12 months are averaged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:  # installed as a package
    from .grid import DTYPES, find_grid_files, read_grid
except ImportError:  # run directly as a file
    from grid import DTYPES, find_grid_files, read_grid


def load(path: Path):
    ext = path.suffix.lstrip(".").upper()
    if ext not in DTYPES:
        sys.exit(f"Unknown extension .{ext} — expected UNF0/1/2/4")
    data = np.fromfile(path, dtype=DTYPES[ext]).astype(np.float64)
    return data, ext


def main():
    ap = argparse.ArgumentParser(prog="unf-view", description="View a UNF file")
    ap.add_argument("file", type=Path)
    ap.add_argument("--layer", type=int, metavar="1-12",
                    help="month layer (12-layer files only); omitted: annual mean")
    ap.add_argument("--stats", action="store_true", help="stats only, no map")
    ap.add_argument("--save", type=Path, metavar="PNG", help="save map as PNG instead of showing")
    ap.add_argument("--csv", type=Path, metavar="CSV", help="export values as cell,value CSV")
    ap.add_argument("--log", action="store_true", help="logarithmic colour scale")
    ap.add_argument("--grid", type=Path, help="folder with GR.UNF2/GC.UNF2 (default: search automatically)")
    args = ap.parse_args()

    data, ext = load(args.file)

    # Find grid files → determine cell count
    grid_dir = args.grid or find_grid_files(args.file.resolve().parent)
    if grid_dir is None:
        sys.exit("GR.UNF2/GC.UNF2 not found — please specify with --grid")
    gr, gc = read_grid(grid_dir)
    ncells = len(gr)

    # Monthly file?
    if len(data) == 12 * ncells:
        data = data.reshape(ncells, 12)
        if args.layer:
            if not 1 <= args.layer <= 12:
                sys.exit("--layer must be 1–12")
            data, label = data[:, args.layer - 1], f"month {args.layer}"
        else:
            data, label = data.mean(axis=1), "annual mean (12 months)"
    elif len(data) == ncells:
        if args.layer:
            sys.exit("File has only one layer — omit --layer")
        label = "static"
    else:
        sys.exit(f"{len(data)} values do not match cell count {ncells} "
                 f"(GAREA etc. are not cell maps)")

    # Statistics
    valid = data[np.isfinite(data)]
    print(f"File:     {args.file.name}  ({ext}, {label})")
    print(f"Cells:    {ncells}")
    print(f"Min/Max:  {valid.min():.6g} / {valid.max():.6g}")
    print(f"Mean:     {valid.mean():.6g}   Median: {np.median(valid):.6g}")
    print(f"Sum:      {valid.sum():.6g}   Zeros: {(valid == 0).sum()} "
          f"({(valid == 0).mean() * 100:.1f} %)")

    if args.csv:
        np.savetxt(args.csv, np.column_stack([np.arange(1, ncells + 1), data]),
                   fmt="%d,%g", header="cell,value", comments="")
        print(f"CSV:      {args.csv}")

    if args.stats and not args.save:
        return

    # Build map (1-based GR/GC → grid)
    nrow, ncol = gr.max(), gc.max()
    grid = np.full((nrow, ncol), np.nan)
    grid[gr - 1, gc - 1] = data

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_grid = grid
    norm = None
    if args.log:
        plot_grid = np.where(grid > 0, grid, np.nan)
        norm = LogNorm(vmin=np.nanmin(plot_grid), vmax=np.nanmax(plot_grid))
    im = ax.imshow(plot_grid, cmap="viridis", norm=norm, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(f"{args.file.name} — {label}")
    ax.set_xlabel("column (GC)")
    ax.set_ylabel("row (GR)")
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"PNG:      {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
