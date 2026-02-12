# Chicago loader: raw on-disk contract (v1 placeholder)

This is the minimal, **deterministic** on-disk input contract consumed by
`motac.loaders.chicago.load_y_obs_matrix`.

## Files

### `y_obs.csv`

- Format: CSV **without a header**.
- Shape: `(n_locations, n_steps)`.
- Semantics:
  - Rows are **location indices**.
    - Row `i` corresponds to location index `i`.
    - The loader preserves file row order exactly (no sorting/reindexing).
  - Columns are **daily** time steps.
    - Column `t` corresponds to time index `t`.
- Values:
  - non-negative integers (counts).

### `mobility.npy` (optional)

- Format: `numpy.save`/`numpy.load` `.npy` file.
- Shape: `(n_locations, n_locations)`.
- If omitted, the loader uses the identity matrix.

## CLI usage

The CLI entry point `motac data chicago-load` reads a tiny JSON config and
prints a small JSON summary (including the loaded `y_obs` shape).

Example config (`chicago_raw.json`):

```json
{
  "path": "/path/to/chicago_raw_v1",
  "mobility_path": null
}
```

Expected directory layout:

```text
/path/to/chicago_raw_v1/
  y_obs.csv
  mobility.npy   # optional
```

Run:

```bash
uv run motac data chicago-load --config chicago_raw.json
```

## Determinism

- Given identical bytes on disk, the loader returns identical `y_obs` and
  mobility arrays.
- Row ordering is defined entirely by the on-disk row order of `y_obs.csv`.
