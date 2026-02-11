from __future__ import annotations

import json

import networkx as nx
import numpy as np
import osmnx as ox

from motac.substrate.builder import SubstrateBuilder, SubstrateConfig


def test_builder_includes_poi_min_travel_time_feature(tmp_path) -> None:
    # Tiny offline graph
    graphml = tmp_path / "tiny.graphml"
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    G.add_node(0, x=-0.1, y=51.5)
    G.add_node(1, x=-0.099, y=51.5)
    G.add_edge(0, 1, key=0, travel_time=60.0, length=100.0)
    G.add_edge(1, 0, key=0, travel_time=60.0, length=100.0)
    ox.save_graphml(G, filepath=graphml)

    # One POI
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
                    }
                ],
            }
        )
    )

    cfg = SubstrateConfig(
        graphml_path=str(graphml),
        cell_size_m=100.0,
        max_travel_time_s=120.0,
        poi_geojson_path=str(poi_path),
        poi_tags={"amenity": ["school"]},
        poi_travel_time_features=True,
    )

    sub = SubstrateBuilder(cfg).build()
    assert sub.poi is not None

    names = sub.poi.feature_names
    assert "poi_min_travel_time_s" in names
    assert "poi_amenity=school_min_travel_time_s" in names

    x = sub.poi.x
    assert x.shape[0] == len(sub.grid.lat)
    # feature columns exist
    assert x.shape[1] == len(names)
    assert np.all(np.isfinite(x))
