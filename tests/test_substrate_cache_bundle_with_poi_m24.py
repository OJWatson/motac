from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from motac.substrate.builder import SubstrateBuilder, SubstrateConfig


def _bundle_sha256(cache_dir: Path, files: list[str]) -> str:
    h = hashlib.sha256()
    for name in files:
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update((cache_dir / name).read_bytes())
        h.update(b"\x00")
    return h.hexdigest()


def _write_tiny_graphml(path: Path) -> None:
    # Hand-written GraphML to keep bytes stable across NetworkX/OSMnx versions.
    path.write_text(
        """<?xml version=\"1.0\" encoding=\"utf-8\"?>
<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">
  <key id=\"crs\" for=\"graph\" attr.name=\"crs\" attr.type=\"string\"/>
  <key id=\"x\" for=\"node\" attr.name=\"x\" attr.type=\"double\"/>
  <key id=\"y\" for=\"node\" attr.name=\"y\" attr.type=\"double\"/>
  <key id=\"length\" for=\"edge\" attr.name=\"length\" attr.type=\"double\"/>
  <key id=\"travel_time\" for=\"edge\" attr.name=\"travel_time\" attr.type=\"double\"/>
  <graph edgedefault=\"directed\">
    <data key=\"crs\">EPSG:4326</data>
    <node id=\"0\"><data key=\"x\">-0.1</data><data key=\"y\">51.5</data></node>
    <node id=\"1\"><data key=\"x\">-0.099</data><data key=\"y\">51.5</data></node>
    <edge id=\"0\" source=\"0\" target=\"1\"><data key=\"length\">100.0</data><data key=\"travel_time\">60.0</data></edge>
    <edge id=\"1\" source=\"1\" target=\"0\"><data key=\"length\">100.0</data><data key=\"travel_time\">60.0</data></edge>
  </graph>
</graphml>
""",
        encoding="utf-8",
    )


def test_cache_bundle_includes_poi_and_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    # Stable timestamps for deterministic meta.json.
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")

    graphml = tmp_path / "tiny.graphml"
    _write_tiny_graphml(graphml)

    # GeoJSON with two POIs (same cell); properties used for tag breakouts.
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
                        "properties": {"amenity": "hospital"},
                        "geometry": {"type": "Point", "coordinates": [-0.0999, 51.5]},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    # Use a huge cell size so we only get 1 grid cell. Keeps feature bytes stable.
    cache_dir = tmp_path / "cache"
    cfg = SubstrateConfig(
        graphml_path=str(graphml),
        cell_size_m=100_000.0,
        max_travel_time_s=120.0,
        disable_pois=False,
        poi_geojson_path=str(poi_path),
        poi_tags={"amenity": ["school", "hospital"]},
        poi_travel_time_features=False,
        cache_dir=str(cache_dir),
    )

    sub = SubstrateBuilder(cfg).build()
    assert sub.poi is not None

    assert (cache_dir / "poi.npz").exists()

    meta = json.loads((cache_dir / "meta.json").read_text())
    assert meta["has_poi"] is True
    files = list(meta["bundle_files"])
    assert files == sorted(files)
    assert "poi.npz" in files
    assert meta["bundle_sha256"] == _bundle_sha256(cache_dir, files)

    # Load + sanity check POI matrix.
    poi = np.load(cache_dir / "poi.npz")
    assert poi["x"].shape[0] == 1
    assert "feature_names" in poi

    # Regression: repeat build is byte-for-byte identical.
    cache_dir2 = tmp_path / "cache2"
    cfg2 = SubstrateConfig(
        graphml_path=str(graphml),
        cell_size_m=100_000.0,
        max_travel_time_s=120.0,
        disable_pois=False,
        poi_geojson_path=str(poi_path),
        poi_tags={"amenity": ["school", "hospital"]},
        poi_travel_time_features=False,
        cache_dir=str(cache_dir2),
    )
    SubstrateBuilder(cfg2).build()

    meta2 = json.loads((cache_dir2 / "meta.json").read_text())
    assert meta2["bundle_sha256"] == meta["bundle_sha256"]
    assert meta2["provenance_sha256"] == meta["provenance_sha256"]
    for name in files + ["meta.json"]:
        assert (cache_dir2 / name).read_bytes() == (cache_dir / name).read_bytes()


def test_poi_feature_ordering_is_stable_across_tag_dict_order(tmp_path: Path) -> None:
    graphml = tmp_path / "tiny.graphml"
    _write_tiny_graphml(graphml)

    poi_path = tmp_path / "pois.geojson"
    poi_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"amenity": "school", "shop": "supermarket"},
                        "geometry": {"type": "Point", "coordinates": [-0.1, 51.5]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tags_a = {"shop": True, "amenity": ["school", "hospital"]}
    tags_b = {"amenity": ["hospital", "school"], "shop": True}

    cfg_a = SubstrateConfig(
        graphml_path=str(graphml),
        cell_size_m=100_000.0,
        max_travel_time_s=120.0,
        poi_geojson_path=str(poi_path),
        poi_tags=tags_a,
        poi_travel_time_features=False,
    )
    cfg_b = SubstrateConfig(
        graphml_path=str(graphml),
        cell_size_m=100_000.0,
        max_travel_time_s=120.0,
        poi_geojson_path=str(poi_path),
        poi_tags=tags_b,
        poi_travel_time_features=False,
    )

    sub_a = SubstrateBuilder(cfg_a).build()
    sub_b = SubstrateBuilder(cfg_b).build()

    assert sub_a.poi is not None and sub_b.poi is not None
    assert sub_a.poi.feature_names == sub_b.poi.feature_names
    assert sub_a.poi.feature_names == [
        "poi_count",
        "amenity=hospital",
        "amenity=school",
        "shop",
    ]
