#!/usr/bin/env python3
"""
unfold — WaterGAP UNF (continental grids) to GeoTIFF and/or GeoPackage.

Unfolds WaterGAP UNF binary files into georeferenced maps.
One tool, two interfaces:
  • without arguments / with --gui  → graphical interface (Tkinter)
  • with files as arguments         → command line (batch-capable)

Outputs:
  • GeoTIFF  — raster, 12 monthly bands (or 1 band with --mean / static).
  • GeoPackage — one polygon per model cell, values as attribute columns
    (m01…m12 + mean, or value). Directly styleable in QGIS; column GCRC_EU
    also allows joining with other tables. NO existing cell grid is needed —
    the polygons are built from GR/GC + georeference.

The grid (corner + cell size) is built in for 'eu'; other continents via
--continent, --mother <shapefile> or --west/--north/--cellsize.

Examples (CLI):
  unfold unf/G_SURFACE_RUNOFF/G_SURFACE_RUNOFF_2011.12.UNF0    # → .gpkg (eu)
  unfold file.UNF0 --format both --mean
  unfold file.UNF0 --mother mother_af_gcrc.shp                # other grid from shapefile
  unfold file.UNF0 --west -180 --north 83.5 --cellsize 0.08333   # grid manually
  unfold unf/OTHER_UNF_FILES/*.UNF*                            # batch
  unfold                                                       # GUI
  python -m unfold file.UNF0                                   # without installation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:  # installed as a package
    from .grid import (CONTINENTS, CRS, NODATA, GridSpec, dims_warning,
                       expected_continent, find_grid_files, load_grids_file,
                       read_grid, read_layers, resolve_spec)
except ImportError:  # run directly as a file
    from grid import (CONTINENTS, CRS, NODATA, GridSpec, dims_warning,
                      expected_continent, find_grid_files, load_grids_file,
                      read_grid, read_layers, resolve_spec)


# ── Writers ───────────────────────────────────────────────────────────────────

def write_geotiff(out: Path, layers: np.ndarray, gr, gc, spec: GridSpec) -> Path:
    """Write raster (NoData −9999), 1 band per layer."""
    import rasterio

    _, nb = layers.shape
    nrow, ncol = gr.max(), gc.max()
    grid = np.full((nb, nrow, ncol), NODATA, dtype=np.float32)
    for b in range(nb):
        grid[b, gr - 1, gc - 1] = layers[:, b]

    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out, "w", driver="GTiff", height=nrow, width=ncol, count=nb,
        dtype="float32", crs=CRS,
        transform=spec.transform(),
        nodata=NODATA, compress="deflate",
    ) as dst:
        dst.write(grid)
        if nb == 12:
            dst.descriptions = tuple(f"month {m}" for m in range(1, 13))
    return out


def write_geopackage(out: Path, layers: np.ndarray, gr, gc, labels, layer_name: str,
                     spec: GridSpec) -> Path:
    """Write vector grid: one cell polygon per cell, values as columns."""
    import geopandas as gpd
    import pandas as pd
    import shapely

    ncells = layers.shape[0]
    # Cell edges directly from 1-based GR/GC + georeference (no cell grid needed)
    xmin = spec.west + (gc - 1) * spec.cell
    ymax = spec.north - (gr - 1) * spec.cell
    geom = shapely.box(xmin, ymax - spec.cell, xmin + spec.cell, ymax)

    attrs = {"GCRC_EU": np.arange(1, ncells + 1, dtype=np.int32),
             "GR": gr.astype(np.int32), "GC": gc.astype(np.int32)}
    vals = np.where(layers == NODATA, np.nan, layers)
    for i, name in enumerate(labels):
        attrs[name] = vals[:, i]
    if len(labels) == 12:  # convenient annual-mean column for styling
        attrs["mean"] = np.nanmean(vals, axis=1)

    gdf = gpd.GeoDataFrame(pd.DataFrame(attrs), geometry=geom, crs=CRS)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, layer=layer_name, driver="GPKG")
    return out


def _layer_name(path: Path) -> str:
    """File name → valid, lowercase GPKG layer name."""
    s = "".join(c.lower() if c.isalnum() else "_" for c in path.stem)
    s = s.strip("_") or "layer"
    return s if s[0].isalpha() else "l_" + s


# ── Orchestrating one file ────────────────────────────────────────────────────

def convert_file(path: Path, grid_dir: Path, formats, mean: bool, spec: GridSpec,
                 out_dir: Path | None = None, out_path: Path | None = None,
                 log=print) -> list[Path]:
    """Convert one UNF file to the requested formats. Returns output paths."""
    gr, gc = read_grid(grid_dir)
    layers, labels = read_layers(path, len(gr), mean)
    nb = layers.shape[1]
    log(f"▸ {path.name}  ({nb} layers: {', '.join(labels)})")

    written: list[Path] = []
    for fmt in formats:
        suffix = ".tif" if fmt == "tif" else ".gpkg"
        if out_path is not None:
            dest = out_path
        else:
            base = (out_dir / path.name) if out_dir else path
            dest = base.with_suffix(suffix)
        if fmt == "tif":
            written.append(write_geotiff(dest, layers, gr, gc, spec))
        else:
            written.append(write_geopackage(dest, layers, gr, gc, labels, _layer_name(path), spec))
        log(f"  ✓ {dest}")
    return written


def resolve_formats(choice: str) -> list[str]:
    return {"tif": ["tif"], "gpkg": ["gpkg"], "both": ["gpkg", "tif"]}[choice]


# ── Command line ──────────────────────────────────────────────────────────────

def run_cli(argv) -> int:
    ap = argparse.ArgumentParser(
        prog="unfold",
        description="UNF → GeoTIFF and/or GeoPackage (WaterGAP continental grids). "
                    "Without files: GUI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("files", nargs="*", type=Path, help="UNF file(s)")
    ap.add_argument("--format", choices=["gpkg", "tif", "both"], default="gpkg",
                    help="output format (default: gpkg)")
    ap.add_argument("--mean", action="store_true",
                    help="combine 12-month files into the annual mean (1 layer)")
    ap.add_argument("-o", "--out", type=Path,
                    help="output file (single input only; format from extension)")
    ap.add_argument("--out-dir", type=Path, help="output folder (otherwise next to the input)")
    ap.add_argument("--grid", type=Path, help="folder with GR.UNF2/GC.UNF2 (default: automatic)")
    ap.add_argument("--gui", action="store_true", help="start the graphical interface")

    grp = ap.add_argument_group("grid georeference")
    grp.add_argument("--continent", help=f"built-in grid (known: {', '.join(sorted(CONTINENTS))}; default: eu)")
    grp.add_argument("--mother", type=Path, help="derive corner from mother shapefile")
    grp.add_argument("--west", type=float, help="longitude of left edge (manual)")
    grp.add_argument("--north", type=float, help="latitude of top edge (manual)")
    grp.add_argument("--cellsize", type=float, help="cell size in degrees (manual, e.g. 0.08333)")
    grp.add_argument("--grids", type=Path, help="JSON with extra grids (name→west/north/cell)")
    args = ap.parse_args(argv)

    if args.gui or not args.files:
        return run_gui()

    # Determine grid georeference
    registry = load_grids_file(args.grids) if args.grids else None
    try:
        spec, source = resolve_spec(continent=args.continent, west=args.west,
                                    north=args.north, cell=args.cellsize,
                                    mother=args.mother, registry=registry)
    except Exception as e:
        ap.error(str(e))
    print(f"  Grid: {source}  (west {spec.west:.4f}, north {spec.north:.4f}, "
          f"cell {spec.cell:.6g}°)")

    formats = resolve_formats(args.format)
    out_path = None
    if args.out is not None:
        if len(args.files) > 1:
            ap.error("-o requires exactly one input file")
        suf = args.out.suffix.lower()
        if suf not in (".tif", ".gpkg"):
            ap.error("-o must end in .tif or .gpkg")
        formats = ["tif" if suf == ".tif" else "gpkg"]
        out_path = args.out

    grid_dir = args.grid or find_grid_files(args.files[0].resolve().parent)
    if grid_dir is None:
        ap.error("GR.UNF2/GC.UNF2 not found — please specify with --grid")

    name = expected_continent(source)
    if name:
        w = dims_warning(name, grid_dir)
        if w:
            print(f"  ⚠ {w}")

    errors = 0
    for f in args.files:
        try:
            convert_file(f, grid_dir, formats, args.mean, spec,
                         out_dir=args.out_dir, out_path=out_path)
        except Exception as e:  # one broken file does not stop the batch
            print(f"  ✗ {f.name}: {e}", file=sys.stderr)
            errors += 1
    return 1 if errors else 0


# ── Graphical interface ───────────────────────────────────────────────────────

def run_gui() -> int:
    import glob
    import os
    import queue
    import subprocess
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    # Optional drag & drop (tkinterdnd2). Without the package the GUI still
    # runs, just without dragging — installable via:  pip install '.[gui]'
    try:
        from tkinterdnd2 import COPY, DND_FILES, TkinterDnD
        root = TkinterDnD.Tk()
        dnd_ok = True
    except Exception:
        root = tk.Tk()
        DND_FILES = COPY = None
        dnd_ok = False

    root.title("Unfold — WaterGAP UNF → GeoTIFF / GeoPackage")
    root.minsize(880, 680)
    root.geometry("980x740")

    style = ttk.Style()
    panel_bg = style.lookup("TFrame", "background") or root.cget("background")

    def _is_dark(color: str) -> bool:
        try:
            r, g, b = root.winfo_rgb(color)
            return (0.299 * r + 0.587 * g + 0.114 * b) / 65535 < 0.5
        except Exception:
            return True

    FG = "#f0f0f0" if _is_dark(panel_bg) else "#1d1d1f"
    SEL_BG = "#3b6ea5"            # selection highlight for input icons
    ICON_W = 128                  # grid width per icon (for wrapping)

    files: list[Path] = []       # input UNF paths
    log_q: "queue.Queue[str]" = queue.Queue()
    grid_var = tk.StringVar()

    # ── File icon (Apple-style: page with dog-ear + extension + name) ─────────
    def file_style(path: Path):
        suf = path.suffix.lower()
        if suf == ".tif":
            return "#3fb950", "TIF"
        if suf == ".gpkg":
            return "#d29922", "GPKG"
        if suf.startswith(".unf"):
            return "#4c8dde", (suf[1:].upper() or "UNF")
        return "#8b949e", (suf[1:].upper() or "DAT")

    def draw_page(cv: "tk.Canvas", color: str, ext: str):
        cv.delete("all")
        x0, y0, x1, y1, fold = 16, 8, 56, 70, 14
        cv.create_polygon(x0, y0, x1 - fold, y0, x1, y0 + fold, x1, y1, x0, y1,
                          fill=color, outline="")
        cv.create_polygon(x1 - fold, y0, x1 - fold, y0 + fold, x1, y0 + fold,
                          fill="#ffffff", stipple="gray25", outline="")
        cv.create_text((x0 + x1) // 2, y1 - 13, text=ext, fill="white",
                       font=("Helvetica", 9, "bold"))

    def make_card(parent: "tk.Widget", path: Path) -> "tk.Frame":
        color, ext = file_style(path)
        card = tk.Frame(parent, bg=panel_bg)
        cv = tk.Canvas(card, width=72, height=78, bg=panel_bg,
                       highlightthickness=0, bd=0)
        draw_page(cv, color, ext)
        cv.pack()
        short = path.name if len(path.name) <= 26 else path.name[:23] + "…"
        lbl = tk.Label(card, text=short, bg=panel_bg, fg=FG, wraplength=112,
                       font=("Helvetica", 10), justify="center")
        lbl.pack()
        card._unf_path = path  # remember for selection/drag
        return card

    # ── Open files from the GUI ───────────────────────────────────────────────
    _GIS_EXT = {".gpkg", ".tif", ".tiff", ".shp", ".geojson", ".asc"}

    def _open_default(path):
        # default app; UNF/GPKG often have none → fall back sensibly.
        try:
            if sys.platform == "darwin":
                r = subprocess.run(["open", str(path)],
                                   capture_output=True, text=True)
                if r.returncode != 0:  # no app claims the file
                    if _qgis and Path(path).suffix.lower() in _GIS_EXT:
                        _open_in(_qgis, path)
                    else:
                        _open_with(path)
            elif sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    def _open_in(app, path):
        try:
            subprocess.Popen(["open", "-a", app, str(path)])
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    def _open_with(path):
        if sys.platform == "darwin":
            app = filedialog.askopenfilename(
                title="Open with — choose app", initialdir="/Applications",
                filetypes=[("Applications", "*.app"), ("All", "*.*")])
            if app:
                _open_in(app, path)
        else:
            messagebox.showinfo(
                "Open with", "Please open via the system file manager.")

    def _reveal(path):
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])
            elif sys.platform.startswith("win"):
                subprocess.Popen(["explorer", "/select,", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(Path(path).parent)])
        except Exception as e:
            messagebox.showerror("Reveal failed", str(e))

    # QGIS, if present, as a handy shortcut (GIS files)
    _qgis = None
    if sys.platform == "darwin":
        _hits = sorted(glob.glob("/Applications/QGIS*.app"))
        _qgis = _hits[-1] if _hits else None
    _reveal_label = "Reveal in Finder" if sys.platform == "darwin" else "Show in folder"

    class FileArea:
        """File area with switchable view (icons/table).
        Optional: drop target (input), drag source (output), multi-select.
        Open files by double-click, right-click for context menu."""

        def __init__(self, parent, *, selectable=False, drop_cb=None,
                     drag_out=False, placeholder=""):
            self.selectable = selectable
            self.drag_out = drag_out
            self.placeholder = placeholder
            self.order: list[Path] = []
            self._sel: set[Path] = set()
            self.mode = "icons"
            self._cards: dict[Path, tk.Frame] = {}
            self._iid: dict[str, Path] = {}
            self.frame = tk.Frame(parent, bg=panel_bg)
            self.frame.rowconfigure(0, weight=1)
            self.frame.columnconfigure(0, weight=1)
            self._build_icons()
            self._build_table()
            if drop_cb and dnd_ok:
                for w in (self.canvas, self.holder, self._ph, self.tree):
                    w.drop_target_register(DND_FILES)
                    w.dnd_bind("<<Drop>>",
                               lambda e: drop_cb(root.tk.splitlist(e.data)))
            self.set_mode("icons")

        # ── Icon view ────────────────────────────────────────────────────────
        def _build_icons(self):
            self.icon_wrap = tk.Frame(self.frame, bg=panel_bg)
            self.canvas = tk.Canvas(self.icon_wrap, bg=panel_bg,
                                    highlightthickness=0, bd=0)
            vsb = ttk.Scrollbar(self.icon_wrap, orient="vertical",
                                command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=vsb.set)
            vsb.pack(side="right", fill="y")
            self.canvas.pack(side="left", fill="both", expand=True)
            self.holder = tk.Frame(self.canvas, bg=panel_bg)
            self._win = self.canvas.create_window((0, 0), window=self.holder, anchor="nw")
            # placeholder floats centred over the canvas (place instead of grid)
            self._ph = tk.Label(self.canvas, text=self.placeholder, bg=panel_bg,
                                fg="gray", justify="center", wraplength=240)
            self.canvas.bind("<Configure>", self._on_canvas_configure)
            self.canvas.bind("<MouseWheel>",
                             lambda e: self.canvas.yview_scroll(-1 * int(e.delta), "units"))

        def _on_canvas_configure(self, e):
            self.canvas.itemconfigure(self._win, width=e.width)
            self._relayout(e.width)

        def _relayout(self, width=None):
            w = int(width or self.canvas.winfo_width())
            cols = max(1, w // ICON_W)
            for i, p in enumerate(self.order):
                card = self._cards.get(p)
                if card:
                    card.grid_configure(row=i // cols, column=i % cols,
                                        padx=8, pady=8, sticky="n")
            self.holder.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        def _render_icons(self):
            for c in self._cards.values():
                c.destroy()
            self._cards.clear()
            if not self.order:
                self._ph.place(relx=0.5, rely=0.5, anchor="center")
            else:
                self._ph.place_forget()
                for p in self.order:
                    card = make_card(self.holder, p)
                    self._cards[p] = card
                    self._wire_card(p, card)
            self._relayout()

        def _wire_card(self, path, card):
            widgets = [card, *card.winfo_children()]
            for w in widgets:
                w.bind("<Double-Button-1>", lambda e, p=path: _open_default(p))
                w.bind("<Button-2>", lambda e, p=path: self._popup(e, p))
                w.bind("<Button-3>", lambda e, p=path: self._popup(e, p))
                w.bind("<Control-Button-1>", lambda e, p=path: self._popup(e, p))
            if self.drag_out and dnd_ok:
                for w in widgets:
                    w.drag_source_register(1, DND_FILES)
                    w.dnd_bind("<<DragInitCmd>>",
                               lambda e, p=path: (COPY, DND_FILES, str(p)))
            if self.selectable:
                for w in widgets:
                    w.bind("<Button-1>", lambda e, p=path: self._toggle(p))
                if path in self._sel:
                    self._paint(card, True)

        def _popup(self, e, path):
            m = tk.Menu(self.frame, tearoff=0)
            m.add_command(label="Open", command=lambda: _open_default(path))
            if _qgis:
                m.add_command(label="Open in QGIS",
                              command=lambda: _open_in(_qgis, path))
            m.add_command(label="Open with…", command=lambda: _open_with(path))
            m.add_separator()
            m.add_command(label=_reveal_label, command=lambda: _reveal(path))
            try:
                m.tk_popup(e.x_root, e.y_root)
            finally:
                m.grab_release()

        def _toggle(self, path):
            on = path not in self._sel
            (self._sel.add if on else self._sel.discard)(path)
            card = self._cards.get(path)
            if card:
                self._paint(card, on)

        def _paint(self, card, on):
            bg = SEL_BG if on else panel_bg
            for w in (card, *card.winfo_children()):
                try:
                    w.configure(bg=bg)
                except tk.TclError:
                    pass

        # ── Table view ───────────────────────────────────────────────────────
        def _build_table(self):
            self.table_wrap = tk.Frame(self.frame, bg=panel_bg)
            cols = ("name", "typ", "ordner")
            self.tree = ttk.Treeview(self.table_wrap, columns=cols,
                                     show="headings", selectmode="extended")
            for c, t, w, stretch in (("name", "Name", 200, True),
                                     ("typ", "Type", 64, False),
                                     ("ordner", "Folder", 180, True)):
                self.tree.heading(c, text=t)
                self.tree.column(c, width=w, anchor="w", stretch=stretch)
            self.tree.tag_configure("unf", foreground="#79a8ec")
            self.tree.tag_configure("tif", foreground="#6cc77f")
            self.tree.tag_configure("gpkg", foreground="#e2b755")
            vsb = ttk.Scrollbar(self.table_wrap, orient="vertical",
                                command=self.tree.yview)
            self.tree.configure(yscrollcommand=vsb.set)
            vsb.pack(side="right", fill="y")
            self.tree.pack(side="left", fill="both", expand=True)
            if self.selectable:
                self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
            if self.drag_out and dnd_ok:
                self.tree.drag_source_register(1, DND_FILES)
                self.tree.dnd_bind("<<DragInitCmd>>", self._tree_drag_init)
            self.tree.bind("<Double-Button-1>", self._tree_open)
            self.tree.bind("<Button-2>", self._tree_menu)
            self.tree.bind("<Button-3>", self._tree_menu)
            self.tree.bind("<Control-Button-1>", self._tree_menu)

        def _tree_path_at(self, e):
            return self._iid.get(self.tree.identify_row(e.y))

        def _tree_open(self, e):
            p = self._tree_path_at(e)
            if p:
                _open_default(p)

        def _tree_menu(self, e):
            iid = self.tree.identify_row(e.y)
            if iid:
                self.tree.selection_set(iid)
                self._popup(e, self._iid[iid])

        def _on_tree_select(self, e):
            self._sel = {self._iid[i] for i in self.tree.selection() if i in self._iid}

        def _tree_drag_init(self, e):
            paths = ([self._iid[i] for i in self.tree.selection() if i in self._iid]
                     or list(self.order))
            if not paths:
                return None
            return (COPY, DND_FILES, [str(p) for p in paths])

        def _render_table(self):
            self.tree.delete(*self.tree.get_children())
            self._iid.clear()
            for p in self.order:
                suf = p.suffix.lower()
                tag = "tif" if suf == ".tif" else "gpkg" if suf == ".gpkg" else "unf"
                iid = self.tree.insert("", "end", tags=(tag,),
                                       values=(p.name, file_style(p)[1], str(p.parent)))
                self._iid[iid] = p
            if self.selectable and self._sel:
                restore = [i for i, p in self._iid.items() if p in self._sel]
                if restore:
                    self.tree.selection_set(restore)

        # ── shared ───────────────────────────────────────────────────────────
        def set_mode(self, mode):
            self.mode = mode
            if mode == "table":
                self.icon_wrap.grid_remove()
                self.table_wrap.grid(row=0, column=0, sticky="nsew")
                self._render_table()
            else:
                self.table_wrap.grid_remove()
                self.icon_wrap.grid(row=0, column=0, sticky="nsew")
                self._render_icons()

        def set_files(self, paths):
            self.order = list(paths)
            self._sel = {p for p in self._sel if p in self.order}
            if self.mode == "table":
                self._render_table()
            else:
                self._render_icons()

        def get_selected(self):
            if self.mode == "table":
                return [self._iid[i] for i in self.tree.selection() if i in self._iid]
            return [p for p in self.order if p in self._sel]

        def clear_selection(self):
            self._sel.clear()
            if self.mode == "table":
                sel = self.tree.selection()
                if sel:
                    self.tree.selection_remove(sel)
            else:
                for card in self._cards.values():
                    self._paint(card, False)

    # ── Input/output logic ────────────────────────────────────────────────────
    def add_paths(paths) -> int:
        added = 0
        for p in paths:
            pp = Path(p)
            if pp.is_dir() or not pp.suffix.upper().startswith(".UNF"):
                continue
            if pp not in files:
                files.append(pp)
                added += 1
        if added:
            in_area.set_files(files)
        if files and not grid_var.get():
            g = find_grid_files(files[0].resolve().parent)
            if g:
                grid_var.set(str(g))
        return added

    def add_files():
        add_paths(filedialog.askopenfilenames(
            title="Select UNF files",
            filetypes=[("UNF files", "*.UNF0 *.UNF1 *.UNF2 *.UNF4"), ("All", "*.*")],
        ))

    def remove_selected():
        sel = in_area.get_selected()
        targets = sel if sel else list(files)  # nothing selected → clear all
        for p in targets:
            if p in files:
                files.remove(p)
        in_area.clear_selection()
        in_area.set_files(files)

    # ── Layout: view switcher + two panels side by side ───────────────────────
    pad = {"padx": 10, "pady": 4}
    main = ttk.Frame(root, padding=10)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=1)
    main.rowconfigure(1, weight=1)

    def hint(parent, text, **grid):
        """Small, muted explanatory line under a field."""
        lbl = ttk.Label(parent, text=text, foreground="gray",
                        wraplength=560, justify="left")
        lbl.grid(sticky="w", **grid)
        return lbl

    # Switcher icons ⇄ table (applies to both panels)
    view_var = tk.StringVar(value="table")

    def apply_view():
        in_area.set_mode(view_var.get())
        out_area.set_mode(view_var.get())

    bar = ttk.Frame(main)
    bar.grid(row=0, column=0, sticky="e", pady=(0, 4))
    ttk.Label(bar, text="View:").pack(side="left", padx=(0, 6))
    ttk.Radiobutton(bar, text="Icons", value="icons", variable=view_var,
                    style="Toolbutton", command=apply_view).pack(side="left")
    ttk.Radiobutton(bar, text="Table", value="table", variable=view_var,
                    style="Toolbutton", command=apply_view).pack(side="left")

    panels = ttk.Frame(main)
    panels.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
    panels.columnconfigure(0, weight=1, uniform="pan")
    panels.columnconfigure(1, weight=1, uniform="pan")
    panels.rowconfigure(0, weight=1)

    # Input panel
    in_frame = ttk.LabelFrame(panels, text="Input (UNF)", padding=6)
    in_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    in_frame.rowconfigure(1, weight=1)
    in_frame.columnconfigure(0, weight=1)
    in_tip = ('Drop files here or use “Add…”' if dnd_ok
              else 'Use “Add…” to load UNF files')
    ttk.Label(in_frame, text=in_tip, foreground="gray").grid(row=0, column=0, sticky="w")
    in_area = FileArea(in_frame, selectable=True, drop_cb=add_paths,
                       placeholder='No files yet.\nDrop here or use “Add…”.')
    in_area.frame.grid(row=1, column=0, sticky="nsew", pady=4)
    in_btns = ttk.Frame(in_frame)
    in_btns.grid(row=2, column=0, sticky="w")
    ttk.Button(in_btns, text="Add…", command=add_files).pack(side="left")
    ttk.Button(in_btns, text="Remove", command=remove_selected).pack(side="left", padx=6)

    # Output panel
    out_frame = ttk.LabelFrame(panels, text="Output", padding=6)
    out_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
    out_frame.rowconfigure(1, weight=1)
    out_frame.columnconfigure(0, weight=1)
    out_tip = ('Drag results into a folder to save' if dnd_ok
               else 'Results appear here')
    ttk.Label(out_frame, text=out_tip, foreground="gray").grid(row=0, column=0, sticky="w")
    out_area = FileArea(out_frame, drag_out=True,
                        placeholder='Results appear after “Start”.')
    out_area.frame.grid(row=1, column=0, sticky="nsew", pady=4)
    apply_view()  # initial view per view_var (table)

    # ── Optionen ─────────────────────────────────────────────────────────────
    opt = ttk.LabelFrame(main, text="Options", padding=8)
    opt.grid(row=2, column=0, sticky="ew", **pad)
    opt.columnconfigure(1, weight=1)

    ttk.Label(opt, text="Format:").grid(row=0, column=0, sticky="w")
    fmt_var = tk.StringVar(value="gpkg")
    fmt_row = ttk.Frame(opt)
    fmt_row.grid(row=0, column=1, sticky="w")
    for txt, val in [("GeoPackage", "gpkg"), ("GeoTIFF", "tif"), ("Both", "both")]:
        ttk.Radiobutton(fmt_row, text=txt, value=val, variable=fmt_var).pack(side="left", padx=(0, 10))

    mean_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(opt, text="Combine 12 months into annual mean",
                    variable=mean_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

    ttk.Label(opt, text="GR/GC folder:").grid(row=2, column=0, sticky="w", pady=(6, 0))
    grid_row = ttk.Frame(opt)
    grid_row.grid(row=2, column=1, sticky="ew", pady=(6, 0))
    grid_row.columnconfigure(0, weight=1)
    ttk.Entry(grid_row, textvariable=grid_var).grid(row=0, column=0, sticky="ew")
    ttk.Button(grid_row, text="…",
               command=lambda: grid_var.set(filedialog.askdirectory(title="Folder with GR.UNF2/GC.UNF2") or grid_var.get())
               ).grid(row=0, column=1, padx=(4, 0))
    hint(opt, "GR.UNF2/GC.UNF2 map each value to its grid row/column. "
              "Found automatically next to the input (change only if needed).",
         row=3, column=1, pady=(2, 0))

    out_var = tk.StringVar()
    ttk.Label(opt, text="Output folder:").grid(row=4, column=0, sticky="w", pady=(6, 0))
    out_row = ttk.Frame(opt)
    out_row.grid(row=4, column=1, sticky="ew", pady=(6, 0))
    out_row.columnconfigure(0, weight=1)
    ttk.Entry(out_row, textvariable=out_var).grid(row=0, column=0, sticky="ew")
    ttk.Button(out_row, text="…",
               command=lambda: out_var.set(filedialog.askdirectory(title="Output folder (empty = next to input)") or out_var.get())
               ).grid(row=0, column=1, padx=(4, 0))
    hint(opt, "Where output files go. Empty = next to each input file.",
         row=5, column=1, pady=(2, 0))

    ttk.Label(opt, text="Continent:").grid(row=6, column=0, sticky="w", pady=(6, 0))
    cont_var = tk.StringVar(value="eu")
    ttk.Combobox(opt, textvariable=cont_var, values=sorted(CONTINENTS),
                 state="readonly", width=12).grid(row=6, column=1, sticky="w", pady=(6, 0))

    mother_var = tk.StringVar()
    ttk.Label(opt, text="Other continent\n(mother shapefile):").grid(row=7, column=0, sticky="w", pady=(6, 0))
    mother_row = ttk.Frame(opt)
    mother_row.grid(row=7, column=1, sticky="ew", pady=(6, 0))
    mother_row.columnconfigure(0, weight=1)
    ttk.Entry(mother_row, textvariable=mother_var).grid(row=0, column=0, sticky="ew")
    ttk.Button(mother_row, text="…",
               command=lambda: mother_var.set(filedialog.askopenfilename(
                   title="Mother shapefile (.shp)",
                   filetypes=[("Shapefile", "*.shp"), ("All", "*.*")]) or mother_var.get())
               ).grid(row=0, column=1, padx=(4, 0))
    hint(opt, "Only for a grid not in the continent list above.",
         row=8, column=1, pady=(2, 0))

    # 3) Start + Log
    start_btn = ttk.Button(main, text="Start")
    start_btn.grid(row=3, column=0, sticky="ew", **pad)

    log_frame = ttk.LabelFrame(main, text="Log", padding=6)
    log_frame.grid(row=4, column=0, sticky="ew", **pad)
    log_frame.columnconfigure(0, weight=1)
    log_widget = tk.Text(log_frame, height=6, wrap="word", state="disabled",
                         takefocus=0, cursor="arrow")
    log_widget.grid(row=0, column=0, sticky="ew")
    log_sb = ttk.Scrollbar(log_frame, orient="vertical", command=log_widget.yview)
    log_sb.grid(row=0, column=1, sticky="ns")
    log_widget.configure(yscrollcommand=log_sb.set)
    for _seq in ("<Button-1>", "<B1-Motion>", "<Double-Button-1>", "<Button-2>", "<Button-3>"):
        log_widget.bind(_seq, lambda e: "break")  # read-only, no click-in

    def gui_log(msg: str):
        log_q.put(msg)

    def drain_log():
        while not log_q.empty():
            log_widget.configure(state="normal")
            log_widget.insert("end", log_q.get() + "\n")
            log_widget.see("end")
            log_widget.configure(state="disabled")
        root.after(120, drain_log)

    def worker(file_list, grid_dir, formats, mean, out_dir, continent, mother):
        try:  # resolve grid (reading shapefile if needed) in the background
            spec, source = resolve_spec(continent=continent, mother=mother)
        except Exception as e:
            gui_log(f"  ✗ Grid: {e}")
            root.after(0, lambda: start_btn.configure(state="normal"))
            return
        gui_log(f"Grid: {source}  (west {spec.west:.4f}, north {spec.north:.4f}, "
                f"cell {spec.cell:.6g}°)")
        name = expected_continent(source)
        if name:
            w = dims_warning(name, grid_dir)
            if w:
                gui_log(f"  ⚠ {w}")
        ok = err = 0
        produced: list[Path] = []
        for f in file_list:
            try:
                produced += convert_file(f, grid_dir, formats, mean, spec,
                                         out_dir=out_dir, log=gui_log)
                ok += 1
            except Exception as e:
                gui_log(f"  ✗ {f.name}: {e}")
                err += 1
        gui_log(f"\nDone: {ok} ok, {err} errors.")
        root.after(0, lambda p=produced: out_area.set_files(p))
        root.after(0, lambda: start_btn.configure(state="normal"))

    def on_start():
        if not files:
            messagebox.showwarning("No files", "Please add UNF files first.")
            return
        grid_dir = Path(grid_var.get()) if grid_var.get() else find_grid_files(files[0].resolve().parent)
        if not grid_dir or not (Path(grid_dir) / "GR.UNF2").exists():
            messagebox.showerror("GR/GC missing", "Folder with GR.UNF2/GC.UNF2 not found — please choose one.")
            return
        out_dir = Path(out_var.get()) if out_var.get() else None
        mother = Path(mother_var.get()) if mother_var.get() else None
        out_area.set_files([])  # clear previous results
        start_btn.configure(state="disabled")
        threading.Thread(
            target=worker,
            args=(list(files), Path(grid_dir), resolve_formats(fmt_var.get()),
                  mean_var.get(), out_dir, cont_var.get(), mother),
            daemon=True,
        ).start()

    start_btn.configure(command=on_start)
    root.after(120, drain_log)
    root.mainloop()
    return 0


def main() -> int:
    return run_cli(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
