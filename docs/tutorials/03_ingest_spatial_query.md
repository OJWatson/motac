# End-to-end: ingest → spatial join → query

This tutorial shows the smallest end-to-end workflow:

1. **Ingest** raw events into motac's canonical events table.
2. **Assign** each event to a **regular grid cell** (a "spatial join").
3. **Query** the resulting table (filter + aggregate).

The goal is to make the data contract concrete: once your events are in the canonical schema and have a `cell_id`, you can feed them into downstream models.

## 1) Raw input: newline-delimited JSON (JSONL)

A raw stream is a JSON object per line.

Required keys:
- `t` (YYYY-MM-DD)
- `lat`, `lon`

Optional keys:
- `event_id`, `value` (defaults to 1), `mark`, `meta` (mapping), `cell_id`

```python
from __future__ import annotations

import json
from pathlib import Path

raw_path = Path("events.jsonl")
rows = [
    {"t": "2020-01-02", "lat": 51.50, "lon": -0.10, "value": 2, "event_id": "a"},
    {"t": "2020-01-03", "lat": 51.52, "lon": -0.12, "mark": "x", "meta": {"src": "demo"}},
]
raw_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
```

## 2) Ingest to the canonical Arrow table (and optionally write Parquet)

```python
import pyarrow as pa

from motac.ingestion import (
    ingest_jsonl_to_canonical_table,
    validate_canonical_events_table,
    write_canonical_events_parquet,
)

tbl = ingest_jsonl_to_canonical_table(raw_path)
validate_canonical_events_table(tbl)
assert isinstance(tbl, pa.Table)

# Optional: persist to Parquet (recommended for larger datasets)
write_canonical_events_parquet(tbl, "events.parquet")
```

At this point you have a strict, predictable schema (see `motac.ingestion.validate_canonical_events_table`).

## 3) Build a regular grid and assign `cell_id`

You can build a tiny regular grid from lon/lat bounds, then map each event to a `cell_id`.

```python
import numpy as np
import pyarrow as pa

from motac.spatial.grid_builder import LonLatBounds, build_regular_grid
from motac.spatial.lookup import GridCellLookup

bounds = LonLatBounds(
    lon_min=-0.20,
    lon_max=0.05,
    lat_min=51.45,
    lat_max=51.60,
)

grid = build_regular_grid(bounds=bounds, cell_size_m=2_000.0)
lookup = GridCellLookup.from_grid(grid)

lon = np.asarray(tbl["lon"].to_numpy(zero_copy_only=False), dtype=float)
lat = np.asarray(tbl["lat"].to_numpy(zero_copy_only=False), dtype=float)
cell_id = lookup.lonlat_to_cell_id(lon=lon, lat=lat)

# Convention: -1 means outside the grid; we convert it to nulls in Arrow.
cell_arrow = pa.array([None if int(x) < 0 else int(x) for x in cell_id], type=pa.int32())

# Add/replace the column.
if "cell_id" in tbl.column_names:
    tbl = tbl.drop(["cell_id"]).append_column("cell_id", cell_arrow)
else:
    tbl = tbl.append_column("cell_id", cell_arrow)

# Keep canonical ordering: required cols, then optional in stable order.
cols = ["t", "lat", "lon", "value", "event_id", "cell_id", "mark", "meta_json"]
tbl = tbl.select([c for c in cols if c in tbl.column_names])
```

If you already have a substrate cache directory containing `grid.npz`, you can also map a point from the CLI:

```bash
motac spatial cell-id --grid path/to/cache_dir --lon -0.10 --lat 51.50
```

## 4) Query: filter and aggregate

Here's a minimal example using Arrow compute:

```python
import pyarrow.compute as pc

# Keep only rows that landed inside the grid.
mask_inside = pc.is_valid(tbl["cell_id"])
tbl_inside = tbl.filter(mask_inside)

# Example query: count events per day.
# (We treat `value` as the event weight.)
by_day = tbl_inside.group_by("t").aggregate([("value", "sum")])
print(by_day)
```

Next steps:
- write the joined table back to Parquet,
- build POI features on the same grid (see the POI tutorial),
- fit a model using the canonical dataset interface.
