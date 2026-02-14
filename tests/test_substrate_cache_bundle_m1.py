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


def test_substrate_cache_bundle_and_meta_hash(tmp_path: Path, monkeypatch) -> None:
    # Ensure stable timestamping for meta.json regardless of the surrounding environment.
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")

    graphml = tmp_path / "tiny.graphml"
    _write_tiny_graphml(graphml)

    cache_dir = tmp_path / "cache"
    cfg = SubstrateConfig(
        graphml_path=str(graphml),
        cell_size_m=100.0,
        max_travel_time_s=120.0,
        disable_pois=True,
        cache_dir=str(cache_dir),
    )

    substrate = SubstrateBuilder(cfg).build()
    assert substrate.graphml_path is not None

    # Artefact bundle exists.
    assert (cache_dir / "graph.graphml").exists()
    assert (cache_dir / "grid.npz").exists()
    assert (cache_dir / "neighbours.npz").exists()
    assert (cache_dir / "meta.json").exists()

    meta = json.loads((cache_dir / "meta.json").read_text())
    for k in [
        "cache_format_version",
        "built_at_utc",
        "motac_version",
        "config",
        "config_sha256",
        "graphml_path",
        "has_poi",
        "bundle_sha256",
        "bundle_files",
    ]:
        assert k in meta

    assert meta["has_poi"] is False
    assert meta["built_at_utc"] == "1970-01-01T00:00:00Z"

    # Hash should match a recomputation of the stored config.
    cfg_dict = meta["config"]
    cfg_json = json.dumps(cfg_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected = hashlib.sha256(cfg_json).hexdigest()
    assert meta["config_sha256"] == expected

    # Regression: config hashing must be deterministic for a fixed input.
    # We *can* pin these because the GraphML fixture is hand-written and copied
    # byte-for-byte into the cache.
    assert cfg_dict["graphml_sha256"] == (
        "3313e03844dc7bbeec32c0e34d0bedb981e3dfbf8f288f052859f0fb07db5abe"
    )
    assert meta["config_sha256"] == (
        "6dcd54b6c472f0d361f945a7072bb5f8a788ab6fab9dccc88e7bd18531332404"
    )

    # Bundle hash should match a recomputation over the referenced files.
    files = list(meta["bundle_files"])
    assert files == sorted(files)  # stable ordering
    assert meta["bundle_sha256"] == _bundle_sha256(cache_dir, files)

    # Regression: bundle hash must be present and deterministic for a fixed input.
    assert meta["bundle_sha256"]
    assert meta["bundle_sha256"] == (
        "8b0a063d6cac364e89396dbd0b7fb48bf7b16df42a7fdc989e9eea3313d9521e"
    )

    # Regression: bundle writes are deterministic byte-for-byte.
    cache_dir2 = tmp_path / "cache2"
    cfg2 = SubstrateConfig(
        graphml_path=str(graphml),
        cell_size_m=100.0,
        max_travel_time_s=120.0,
        disable_pois=True,
        cache_dir=str(cache_dir2),
    )
    SubstrateBuilder(cfg2).build()

    meta2 = json.loads((cache_dir2 / "meta.json").read_text())
    assert meta2["bundle_sha256"] == meta["bundle_sha256"]
    for name in files + ["meta.json"]:
        assert (cache_dir2 / name).read_bytes() == (cache_dir / name).read_bytes()

    # Sanity load core arrays.
    grid = np.load(cache_dir / "grid.npz")
    assert grid["lat"].ndim == 1 and grid["lon"].ndim == 1
    assert grid["lat"].shape == grid["lon"].shape
