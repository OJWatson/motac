# Substrate cache bundle (v1)

When `SubstrateConfig.cache_dir` is set, `SubstrateBuilder.build()` writes a **self-contained
bundle** that can be reloaded later without re-downloading OSM data.

The bundle is designed to be:

- **portable** (everything needed is inside `cache_dir/`)
- **versioned** (via `meta.json["cache_format_version"]`)
- **validated** (via a stable `config_sha256` hash)

## Files

A v1 bundle contains:

- `graph.graphml`
- `grid.npz`
- `neighbours.npz`
- `meta.json`
- `poi.npz` (optional; only present when POIs are enabled)

### `graph.graphml`

Road network saved via `osmnx.save_graphml`.

### `grid.npz`

Compressed NumPy archive written by `numpy.savez_compressed`.

Arrays:

- `lat`: `float64`, shape `(n_cells,)` — grid centroid latitudes (EPSG:4326)
- `lon`: `float64`, shape `(n_cells,)` — grid centroid longitudes (EPSG:4326)
- `cell_size_m`: `float64`, shape `(1,)` — grid spacing in metres

### `neighbours.npz`

SciPy sparse matrix written by `scipy.sparse.save_npz`.

- type: CSR (the loader normalizes with `.tocsr()`)
- shape: `(n_cells, n_cells)`
- semantics: `travel_time_s[i, j]` is the shortest-path travel time (seconds) from cell `i` to `j`
  for all `j` reachable within `max_travel_time_s` (plus the diagonal)

### `poi.npz` (optional)

Compressed NumPy archive written by `numpy.savez_compressed`.

Arrays:

- `x`: `float64`, shape `(n_cells, n_features)` — POI feature matrix aligned to the grid
- `feature_names`: `object`, shape `(n_features,)` — strings naming each feature column

### `meta.json`

Human-readable provenance and validation metadata.

Keys:

- `cache_format_version` (int)
- `built_at_utc` (string) — UTC timestamp formatted as `YYYY-MM-DDTHH:MM:SSZ`
- `motac_version` (string)
- `config` (object) — normalized subset of `SubstrateConfig` fields that affect the bundle
- `config_sha256` (string) — SHA-256 over JSON(`config`) with sorted keys and compact separators
- `graphml_path` (string) — path used by the bundle (for cached bundles this points at `cache_dir/graph.graphml`)
- `has_poi` (bool)

## Versioning and compatibility

`SubstrateBuilder` defines `CACHE_FORMAT_VERSION` and embeds it into `meta.json`.
When loading a cache, the builder validates that the stored version matches the
library’s expected version; otherwise it raises a `ValueError`.

This protects against silent misreads if the on-disk format changes.
