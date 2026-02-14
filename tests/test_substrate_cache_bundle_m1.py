from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox

from motac.substrate.builder import SubstrateBuilder, SubstrateConfig


def _bundle_sha256(cache_dir: Path, files: list[str]) -> str:
    h = hashlib.sha256()
    for name in files:
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update((cache_dir / name).read_bytes())
        h.update(b"\x00")
    return h.hexdigest()


def test_substrate_cache_bundle_and_meta_hash(tmp_path: Path) -> None:
    # Build a tiny offline graph.
    graphml = tmp_path / "tiny.graphml"
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    G.add_node(0, x=-0.1, y=51.5)
    G.add_node(1, x=-0.099, y=51.5)
    # Provide travel_time so builder doesn't need speed inference.
    G.add_edge(0, 1, key=0, travel_time=60.0, length=100.0)
    G.add_edge(1, 0, key=0, travel_time=60.0, length=100.0)
    ox.save_graphml(G, filepath=graphml)

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

    # Hash should match a recomputation of the stored config.
    cfg_dict = meta["config"]
    cfg_json = json.dumps(cfg_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected = hashlib.sha256(cfg_json).hexdigest()
    assert meta["config_sha256"] == expected

    # Bundle hash should match a recomputation over the referenced files.
    files = list(meta["bundle_files"])
    assert files == sorted(files)  # stable ordering
    assert meta["bundle_sha256"] == _bundle_sha256(cache_dir, files)

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
