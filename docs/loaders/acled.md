# ACLED loader: events CSV contract (v1 placeholder)

This is the minimal, **deterministic** on-disk input contract consumed by
`motac.loaders.acled.load_acled_events_csv`.

The implementation is intentionally lightweight (no pandas) and uses a
*placeholder* schema that is sufficient for end-to-end wiring + tests.

## File

### `acled.csv`

- Format: CSV **with a header**.
- Each row represents a single event.
- Required columns:
  - `event_date`: ISO date string `YYYY-MM-DD`
  - `lat`: latitude (float, finite, in `[-90, 90]`)
  - `lon`: longitude (float, finite, in `[-180, 180]`)
  - `fatalities`: non-negative integer (encoded as number; must be integer-valued)

## Output semantics

The loader aggregates events to **daily** counts per inferred location.

- Locations are inferred as the unique `(lon, lat)` pairs present in the file.
- `y_obs` has shape `(n_locations, n_days)`.
- `value` controls the aggregated quantity:
  - `value="events"`: counts events per `(location, day)`
  - `value="fatalities"`: sums fatalities per `(location, day)`

## Mobility (optional)

- Optional `mobility.npy` can be provided via `mobility_path`.
- Format: `numpy.save`/`numpy.load` `.npy` file.
- Shape: `(n_locations, n_locations)`.
- If omitted, the loader uses the identity matrix.

## Example usage

```python
from motac.loaders.acled import load_acled_events_csv

out = load_acled_events_csv(
    path="/path/to/acled.csv",
    mobility_path=None,
    value="events",
)

print(out.y_obs.shape)
print(out.meta["dates"][:3])
```

## Determinism

- Given identical bytes on disk, the loader returns identical `y_obs` and
  mobility arrays.
- Locations are ordered by `numpy.unique` over `(lon, lat)` pairs (lexicographic
  by `lon`, then `lat`).
- Days are ordered by sorting ISO date strings `YYYY-MM-DD`.
