# Substrate

The substrate is the road-constrained spatial scaffold used by models and
simulators:

- a road graph (OSMnx GraphML)
- a regular grid (cell centroids)
- sparse travel-time neighbourhoods between grid cells
- optional POI feature matrix aligned to the grid

## Building

Use `SubstrateBuilder(SubstrateConfig(...)).build()` (or the CLI wrapper) to build
and optionally cache artefacts.

`SubstrateConfig` supports three ways to specify the region:

- `bbox=(north,south,east,west)` via the individual fields `north/south/east/west`
- `place="..."` (OSM place query)
- `graphml_path="..."` (offline / tests)

## Cache artefacts

If `cache_dir` is set, the builder writes a self-contained bundle:

- `graph.graphml`
- `grid.npz` (lat/lon centroids + cell size)
- `neighbours.npz` (CSR sparse matrix of travel times, seconds)
- `meta.json` (provenance + config hash + cache format version)
- `poi.npz` (optional; POI feature matrix)

### Cache format versioning

The cache includes `meta.json["cache_format_version"]`. The loader validates this
against the library’s expected version and raises if they mismatch.

## POI features (M2)

If POIs are enabled, `Substrate.poi` is a `POIFeatures` object with:

- `x`: shape `(n_cells, n_features)`
- `feature_names`: list of feature names, aligned to columns of `x`

By default we always include:

- `poi_count`: total number of POIs assigned to each grid cell

### Tag/value breakout counts

If `SubstrateConfig.poi_tags` is provided, we also add optional breakout count
features based on POI properties:

- If `{"amenity": True}` then we add a feature named `amenity` counting POIs with
  a non-null `amenity` property.
- If `{"amenity": ["cafe", "restaurant"]}` then we add features named
  `amenity=cafe` and `amenity=restaurant`.

This works both for OSM downloads (where those properties are columns in the
GeoDataFrame) and for local GeoJSON inputs (as long as `properties` contain the
relevant keys).

### Example config

```json
{
  "place": "Camden, London, UK",
  "cell_size_m": 250.0,
  "max_travel_time_s": 900.0,
  "poi_tags": {"amenity": ["cafe", "restaurant"]},
  "cache_dir": "./cache/camden"
}
```

### Travel-time-to-nearest-POI

M2 v1 currently provides **count-based** POI features only. If you need
travel-time-to-nearest-POI features, add them as a follow-on feature (and bump
cache format if treated as breaking).

---

## API reference

```{eval-rst}
.. automodule:: motac.substrate
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: motac.substrate.builder
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: motac.substrate.types
   :members:
   :undoc-members:
   :show-inheritance:
```
