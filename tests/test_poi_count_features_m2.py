from __future__ import annotations

import json

import networkx as nx
import numpy as np
import osmnx as ox

from motac.substrate.builder import SubstrateBuilder, SubstrateConfig


def test_builder_poi_count_and_tag_breakouts_single_cell(tmp_path) -> None:
    # Tiny offline graph; use a very large cell_size_m so the grid is 1 cell.
    graphml = tmp_path / "tiny.graphml"
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    G.add_node(0, x=-0.1, y=51.5)
    G.add_node(1, x=-0.099, y=51.5001)
    G.add_edge(0, 1, key=0, travel_time=60.0, length=100.0)
    G.add_edge(1, 0, key=0, travel_time=60.0, length=100.0)
    ox.save_graphml(G, filepath=graphml)

    # Three POIs with amenity tags.
    poi_path = tmp_path / "pois.geojson"
    poi_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"amenity": "school"},
                        "geometry": {"type": "Point", "coordinates": [-0.1, 51.5]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"amenity": "school"},
                        "geometry": {"type": "Point", "coordinates": [-0.0998, 51.50005]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"amenity": "hospital"},
                        "geometry": {"type": "Point", "coordinates": [-0.0999, 51.50002]},
                    },
                ],
            }
        )
    )

    cfg = SubstrateConfig(
        graphml_path=str(graphml),
        cell_size_m=100_000.0,
        max_travel_time_s=120.0,
        poi_geojson_path=str(poi_path),
        poi_tags={"amenity": ["school", "hospital"]},
        poi_travel_time_features=False,
    )

    sub = SubstrateBuilder(cfg).build()
    assert sub.poi is not None

    names = sub.poi.feature_names
    assert "poi_count" in names
    assert "amenity=school" in names
    assert "amenity=hospital" in names

    x = sub.poi.x
    assert x.shape == (1, len(names))

    # All POIs are assigned to the only grid cell.
    col = {n: i for i, n in enumerate(names)}
    assert np.allclose(x[0, col["poi_count"]], 3.0)
    assert np.allclose(x[0, col["amenity=school"]], 2.0)
    assert np.allclose(x[0, col["amenity=hospital"]], 1.0)
