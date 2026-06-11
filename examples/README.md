# Examples

This tool needs **UNF files** plus the grid files `GR.UNF2` and `GC.UNF2`
(located in `OTHER_UNF_FILES/`). Both come from the WorldQual-Lite and WaterGAP
data packages — they are **not** versioned here (too large).

Typical folder structure of the input data:

```
unf/
├── G_SURFACE_RUNOFF/
│   └── G_SURFACE_RUNOFF_2011.12.UNF0
├── G_URBAN_RUNOFF/
├── P_RATE_TON_KM2/
└── OTHER_UNF_FILES/
    ├── GR.UNF2
    └── GC.UNF2
```

Quick test once the data is available locally:

```bash
# GeoPackage of a monthly file (GR/GC found automatically)
unfold unf/G_SURFACE_RUNOFF/G_SURFACE_RUNOFF_2011.12.UNF0 --out-dir ./outputs

# view for comparison
unf-view unf/G_SURFACE_RUNOFF/G_SURFACE_RUNOFF_2011.12.UNF0 --stats
```
