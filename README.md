# unfold

## What is it

WaterGAP **UNF** files → **GeoTIFF** (raster) and/or **GeoPackage** (vector) for GIS.
GUI (drag & drop, icons/table, open in QGIS) and CLI (batch).
Plus `unf-view`: quickly view a UNF file (map/stats/PNG/CSV).

## Install

### Prerequisites

Assumes [Homebrew](https://brew.sh) is installed.

```bash
brew install git
brew install python
brew install python-tk
```

### Unfold

```bash
git clone https://github.com/Leon-Muehlenbruch/unfold.git
cd unfold
python3 -m venv .venv && source .venv/bin/activate
pip install '.[gui]'              # installs dependencies + the unfold / unf-view commands (with drag & drop)
```

## What data

You need the **UNF file(s)** (`*.UNF0/1/2/4`) plus the grid files
**`GR.UNF2` + `GC.UNF2`** (row/column per cell, usually in `OTHER_UNF_FILES/`).
The grid files are searched automatically next to the input; otherwise pass `--grid`.

All six WaterGAP continents are built in: `--continent {af,as,au,eu,na,sa}` (default `eu`).
For a grid outside these, use `--mother <shp>` or `--west/--north/--cellsize`.

## Run

```bash
unfold                                   # GUI
unfold file.UNF0                         # → .gpkg (eu)
unfold file.UNF0 --format both --mean    # GeoTIFF + GeoPackage, annual mean
unfold *.UNF0 --format tif --out-dir ./tif --continent af   # batch
unf-view file.UNF0                       # view
```

| Option | Meaning |
|---|---|
| `--format {gpkg,tif,both}` | output format (default `gpkg`) |
| `--mean` | 12 months → annual mean |
| `-o / --out-dir` | output file / output folder (otherwise next to input) |
| `--grid` | folder with `GR.UNF2`/`GC.UNF2` |
| `--continent / --mother / --west,--north,--cellsize` | grid georeference |

MIT licence ([LICENSE](LICENSE)).
