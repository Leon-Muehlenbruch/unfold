"""Shared georeference and readers for WaterGAP continental grids.

A grid's geolocation comes from two parts:
  • GR.UNF2/GC.UNF2  — which cells exist and their 1-based row/column
                       (indexed by GCRC, identical to the UNF data order);
  • a GridSpec       — top-left corner (west/north) + cell size in degrees.

Only the GridSpec is grid-specific. For the EU grid it is built in and
validated (corner coordinates from mother_eu, sample Möhne GCRC 81472); other
continents can be derived from their mother shapefile via
`spec_from_shapefile()`, registered via grids.json, or given directly via
west/north/cell size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Data type from the UNF extension (big-endian, as written by WaterGAP)
DTYPES = {"UNF0": ">f4", "UNF1": "i1", "UNF2": ">i2", "UNF4": ">i4"}

NODATA = -9999.0
CRS = "EPSG:4326"

# Cell sizes of the WaterGAP grid types (degrees)
ARCMIN5 = 1.0 / 12.0   # 5 arc-minutes (continental grids eu/af/as/au/na/sa)
DEG_HALF = 0.5         # 0.5°  (global grids wg2/wa/clm)
DEG_QUARTER = 0.25     # 0.25° (global grid clm025)

# Expected (cell count, nrow, ncol) per WaterGAP grid — from the model (config.py).
# Used as a sanity check: does GR/GC match the chosen continent?
CONTINENT_DIMS: dict[str, tuple[int, int, int]] = {
    "eu": (180721, 641, 1000),   "af": (371410, 1090, 1237),
    "as": (841703, 1258, 4320),  "au": (109084, 740, 4309),
    "na": (461694, 915, 1519),   "sa": (226852, 824, 1356),
    "wg2": (66896, 360, 720),    "wa": (67420, 360, 720),
    "clm": (70412, 360, 720),    "clm025": (281648, 720, 1440),
}


@dataclass(frozen=True)
class GridSpec:
    """Georeference of a grid: top-left corner + cell size (degrees, EPSG:4326)."""
    west: float       # longitude of the left edge
    north: float      # latitude of the top edge
    cell: float       # cell size in degrees

    def transform(self):
        """rasterio affine transform for from_origin(west, north, cell, cell)."""
        from rasterio.transform import from_origin
        return from_origin(self.west, self.north, self.cell, self.cell)


# Built-in, validated grids. Only 'eu' has a confirmed corner; for other
# continents derive it via spec_from_shapefile() and add it here.
CONTINENTS: dict[str, GridSpec] = {
    "eu": GridSpec(west=-31.0 - 20.0 / 60.0, north=80.0 + 50.0 / 60.0, cell=ARCMIN5),
}


def load_grids_file(path: Path) -> dict[str, GridSpec]:
    """Reads additional grids from a JSON file.

    Format: {"af": {"west": .., "north": .., "cell": ..}, ...}
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for name, d in raw.items():
        out[name] = GridSpec(west=float(d["west"]), north=float(d["north"]),
                             cell=float(d["cell"]))
    return out


# Auto-register the bundled grids derived from the cluster ASC headers
# (af/as/au/na/sa). 'eu' stays the hand-validated code constant (takes priority).
_bundled = Path(__file__).with_name("grids.json")
if _bundled.exists():
    try:
        for _name, _spec in load_grids_file(_bundled).items():
            CONTINENTS.setdefault(_name, _spec)
    except Exception:
        pass


def _snap_value(value: float, cell: float, origin: float, tol: float = 0.25) -> float:
    """Snaps value to the nearest multiple of cell from origin (when close)."""
    snapped = origin + round((value - origin) / cell) * cell
    return snapped if abs(snapped - value) <= tol * cell else value


def spec_from_shapefile(path: Path, snap: bool = True) -> GridSpec:
    """Derives a GridSpec from a mother shapefile/layer.

    Works with cell polygons (e.g. mother_<cont>_gcrc.shp) and with cell
    centres (points). The corner is the top-left edge, the cell size is the
    edge length of one cell.

    With ``snap`` (default), cell size and corner are snapped to the global
    WaterGAP raster (origin −180°/90°, cell sizes 5′/0.25°/0.5°) to compensate
    for coordinate inaccuracies in the shapefile.
    """
    import geopandas as gpd

    g = gpd.read_file(path)
    minx, miny, maxx, maxy = (float(v) for v in g.total_bounds)
    gtype = str(g.geom_type.iloc[0])

    if "Polygon" in gtype:
        gx0, gy0, gx1, gy1 = g.geometry.iloc[0].bounds
        cell = float(gx1 - gx0)
        west, north = minx, maxy
    else:  # points = cell centres → push edges outward by half a cell
        xs = np.unique(np.round(g.geometry.x.to_numpy(), 8))
        dx = np.diff(np.sort(xs))
        cell = float(np.median(dx[dx > 1e-9]))
        west, north = minx - cell / 2.0, maxy + cell / 2.0

    if not cell > 0:
        raise ValueError(f"Could not determine cell size from {Path(path).name}")

    if snap:
        for known in (ARCMIN5, DEG_QUARTER, DEG_HALF, 1.0 / 24.0):
            if abs(cell - known) <= 1e-4:
                cell = known
                break
        west = _snap_value(west, cell, -180.0)
        north = _snap_value(north, cell, 90.0)
    return GridSpec(west=float(west), north=float(north), cell=float(cell))


def resolve_spec(continent: str | None = None,
                 west: float | None = None, north: float | None = None,
                 cell: float | None = None, mother: Path | None = None,
                 registry: dict[str, GridSpec] | None = None):
    """Picks a GridSpec by priority and returns (spec, source).

    Order: explicit west/north/cell size > mother shapefile >
    continent registry > default 'eu'.
    """
    reg = dict(CONTINENTS)
    if registry:
        reg.update(registry)

    manual = [v is not None for v in (west, north, cell)]
    if any(manual):
        if not all(manual):
            raise ValueError("--west, --north and --cellsize must be given together")
        return GridSpec(float(west), float(north), float(cell)), "manual"

    if mother is not None:
        return spec_from_shapefile(mother), f"shapefile:{Path(mother).name}"

    if continent:
        if continent not in reg:
            raise ValueError(
                f"Continent '{continent}' unknown. Known: {', '.join(sorted(reg))}. "
                f"Otherwise use --mother or --west/--north/--cellsize."
            )
        return reg[continent], f"continent:{continent}"

    return reg["eu"], "default:eu"


def find_grid_files(start: Path) -> Path | None:
    """Searches for OTHER_UNF_FILES with GR/GC from the file's folder upward."""
    for base in [start, *start.parents]:
        for cand in [base / "OTHER_UNF_FILES", base / "unf" / "OTHER_UNF_FILES"]:
            if (cand / "GR.UNF2").exists() and (cand / "GC.UNF2").exists():
                return cand
    return None


def read_grid(grid_dir: Path):
    """Reads GR/GC (1-based row/column per cell, indexed by GCRC)."""
    grid_dir = Path(grid_dir)
    gr = np.fromfile(grid_dir / "GR.UNF2", dtype=">i2").astype(int)
    gc = np.fromfile(grid_dir / "GC.UNF2", dtype=">i2").astype(int)
    if len(gr) != len(gc):
        raise ValueError(f"GR ({len(gr)}) and GC ({len(gc)}) have different lengths")
    return gr, gc


def expected_continent(source: str) -> str | None:
    """Derive the continent name from the resolve_spec source for the sanity check."""
    if source.startswith("continent:"):
        return source.split(":", 1)[1]
    if source == "default:eu":
        return "eu"
    return None


def dims_warning(name: str, grid_dir: Path) -> str | None:
    """Warns if GR/GC does not match the expected dimensions of the continent."""
    dims = CONTINENT_DIMS.get(name)
    if not dims:
        return None
    gr, gc = read_grid(grid_dir)
    got = (len(gr), int(gr.max()), int(gc.max()))
    if got != dims:
        return (f"GR/GC does not match continent '{name}': {got[0]} cells / "
                f"{got[1]}×{got[2]} instead of {dims[0]} / {dims[1]}×{dims[2]} "
                f"— possibly wrong continent or wrong GR/GC.")
    return None


def read_layers(path: Path, ncells: int, mean: bool = False):
    """Read UNF → (layers[ncells, nb], band_labels).

    Detects monthly files (12×cell count) and single layers (cell count). With
    ``mean``, 12 months are combined into the annual mean (1 layer).
    """
    path = Path(path)
    ext = path.suffix.lstrip(".").upper()
    if ext not in DTYPES:
        raise ValueError(f"Unknown extension .{ext} — expected UNF0/1/2/4")
    data = np.fromfile(path, dtype=DTYPES[ext]).astype(np.float32)

    if len(data) == 12 * ncells:
        layers = data.reshape(ncells, 12)
        labels = [f"m{m:02d}" for m in range(1, 13)]
        if mean:
            layers = layers.mean(axis=1, keepdims=True)
            labels = ["mean"]
    elif len(data) == ncells:
        if mean:
            raise ValueError("File has only one layer — mean not applicable")
        layers = data.reshape(ncells, 1)
        labels = ["value"]
    else:
        raise ValueError(
            f"{len(data)} values do not match cell count {ncells} "
            f"(GAREA etc. are not cell maps)"
        )
    return layers, labels
