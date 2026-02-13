# Tutorial: POI features (baseline)

This tutorial describes the **baseline Point-Of-Interest (POI) features** that can
be included in a substrate build (Roadmap **M2**).

POI features are computed during substrate building and stored on the returned
`Substrate` as `substrate.poi`.

## What you get

When POIs are enabled, `substrate.poi` is a `POIFeatures` object:

- `x`: a NumPy array with shape `(n_cells, n_features)`
- `feature_names`: list of feature names aligned to the columns of `x`

### Count features

We always include the total count:

- `poi_count`

Optionally, you can request **tag/value breakout** counts using `poi_tags`.

Examples:

- `{"amenity": true}` adds a feature `amenity` counting POIs with a non-null
  `amenity` property.
- `{"amenity": ["cafe", "restaurant"]}` adds `amenity=cafe` and
  `amenity=restaurant`.

### Travel-time-to-nearest features (min travel time)

If `poi_travel_time_features=true`, the builder appends travel-time-based
features derived from the cached sparse travel-time neighbourhoods:

- `poi_min_travel_time_s`: minimum travel time (seconds) from each grid cell to
  *any* cell that contains at least one POI.

If you also specify `poi_tags`, tag/value-specific min travel time features are
added too:

- `poi_<tag>_min_travel_time_s` for `<tag>: true` (e.g. `poi_amenity_min_travel_time_s`)
- `poi_<tag>=<value>_min_travel_time_s` for `<tag>: [values]`
  (e.g. `poi_amenity=school_min_travel_time_s`)

If a POI target is not reachable within the neighbourhood cutoff, the feature is
set to `max_travel_time_s`.

## Configuration knobs (SubstrateConfig)

POI behaviour is controlled by these `SubstrateConfig` fields:

- `disable_pois` (bool, default `false`): disable all POI feature generation.
- `poi_geojson_path` (string, optional): path to a local GeoJSON file providing
  POIs (offline / tests / reproducible experiments).
- `poi_tags` (object/dict, optional): tag/value selection for breakout features.
  Interpreted as:
  - `{"tag": true}`: count all POIs where `properties[tag]` is non-null
  - `{"tag": ["v1", "v2"]}`: count only specific values
- `poi_travel_time_features` (bool, default `false`): add min-travel-time
  features to the nearest POI target(s).

POIs can be sourced either by:

- `poi_geojson_path` (local file), or
- OSM-derived POIs using `poi_tags` together with a region spec (`place` or
  `north/south/east/west` or `graphml_path`).

## Example JSON config

```json
{
  "place": "Camden, London, UK",
  "cell_size_m": 250.0,
  "max_travel_time_s": 900.0,
  "poi_tags": {"amenity": ["cafe", "restaurant"]},
  "poi_travel_time_features": true,
  "cache_dir": "./cache/camden"
}
```

## Inspecting feature names

```python
from motac.substrate.builder import SubstrateBuilder, SubstrateConfig

cfg = SubstrateConfig.from_json("./substrate.json")
sub = SubstrateBuilder(cfg).build()

if sub.poi is None:
    raise RuntimeError("POIs disabled or not available")

print(sub.poi.feature_names)
print(sub.poi.x.shape)
```

## Related reference

- API reference: `docs/api/substrate.md` (includes a compact description of the
  same naming conventions)
