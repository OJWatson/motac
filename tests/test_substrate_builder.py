from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox

from motac.substrate import SubstrateBuilder, SubstrateConfig


def _write_tiny_graphml(path: Path) -> None:
    # three nodes roughly in a line in WGS84
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"

    G.add_node(0, x=-0.1000, y=51.5000)
    G.add_node(1, x=-0.0990, y=51.5000)
    G.add_node(2, x=-0.0980, y=51.5000)

    # 60s per hop
    G.add_edge(0, 1, key=0, travel_time=60.0, length=100.0)
    G.add_edge(1, 2, key=0, travel_time=60.0, length=100.0)
    # bidirectional
    G.add_edge(1, 0, key=0, travel_time=60.0, length=100.0)
    G.add_edge(2, 1, key=0, travel_time=60.0, length=100.0)

    ox.save_graphml(G, filepath=path)


def test_build_grid_neighbours_and_cache(tmp_path: Path) -> None:
    graphml = tmp_path / "tiny.graphml"
    _write_tiny_graphml(graphml)

    # two POIs as GeoJSON points
    poi_path = tmp_path / "pois.geojson"
    poi_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"amenity": "cafe"},
                "geometry": {"type": "Point", "coordinates": [-0.1000, 51.5000]},
            },
            {
                "type": "Feature",
                "properties": {"amenity": "cafe"},
                "geometry": {"type": "Point", "coordinates": [-0.0980, 51.5000]},
            },
        ],
    }
    poi_path.write_text(json.dumps(poi_geojson))

    cache_dir = tmp_path / "cache"
    cfg = SubstrateConfig(
        graphml_path=str(graphml),
        cell_size_m=100.0,
        max_travel_time_s=61.0,  # should reach immediate neighbour only
        poi_geojson_path=str(poi_path),
        cache_dir=str(cache_dir),
    )

    s = SubstrateBuilder(cfg).build()

    assert len(s.grid.lat) >= 1
    mat = s.neighbours.travel_time_s
    assert mat.shape[0] == mat.shape[1] == len(s.grid.lat)
    # diagonal present
    assert mat.diagonal().min() == 0.0

    # POI count feature exists
    assert s.poi is not None
    assert s.poi.x.shape[0] == len(s.grid.lat)
    assert s.poi.x.shape[1] == 1
    assert np.isclose(s.poi.x.sum(), 2.0)

    # cache written
    assert (cache_dir / "graph.graphml").exists()
    assert (cache_dir / "grid.npz").exists()
    assert (cache_dir / "neighbours.npz").exists()
    assert (cache_dir / "meta.json").exists()
    assert (cache_dir / "poi.npz").exists()

    meta = json.loads((cache_dir / "meta.json").read_text())
    assert meta["cache_format_version"] == 1
    assert meta["config_sha256"]

    # load from cache yields same sizes
    s2 = SubstrateBuilder(cfg).build()
    assert len(s2.grid.lat) == len(s.grid.lat)
    assert s2.neighbours.travel_time_s.nnz == s.neighbours.travel_time_s.nnz
    assert s2.poi is not None
    assert np.allclose(s2.poi.x, s.poi.x)


def test_cache_format_version_mismatch_raises(tmp_path: Path) -> None:
    graphml = tmp_path / "tiny.graphml"
    _write_tiny_graphml(graphml)

    cache_dir = tmp_path / "cache"
    cfg = SubstrateConfig(
        graphml_path=str(graphml),
        cell_size_m=100.0,
        max_travel_time_s=61.0,
        disable_pois=True,
        cache_dir=str(cache_dir),
    )

    _ = SubstrateBuilder(cfg).build()

    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["cache_format_version"] = 999
    meta_path.write_text(json.dumps(meta))

    import pytest

    with pytest.raises(ValueError, match="cache format version"):
        _ = SubstrateBuilder(cfg).build()
