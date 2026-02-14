from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner
from typer.main import get_command

from motac.cli import app


def test_substrate_build_cache_bundle_smoke_roundtrip(tmp_path: Path) -> None:
    """End-to-end smoke: CLI build writes a cache bundle and can reload from it.

    The second invocation must succeed even if the original graphml_path is removed,
    demonstrating that the bundle is self-contained.
    """

    graphml = tmp_path / "tiny.graphml"
    graphml.write_text(
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

    cache_dir = tmp_path / "cache"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "graphml_path": str(graphml),
                "cell_size_m": 100.0,
                "max_travel_time_s": 120.0,
                "disable_pois": True,
                "cache_dir": str(cache_dir),
            }
        )
    )

    runner = CliRunner()

    res1 = runner.invoke(get_command(app), ["substrate", "build", "--config", str(cfg_path)])
    assert res1.exit_code == 0, res1.stdout
    assert (cache_dir / "graph.graphml").exists()
    assert (cache_dir / "grid.npz").exists()
    assert (cache_dir / "neighbours.npz").exists()
    assert (cache_dir / "meta.json").exists()

    meta = json.loads((cache_dir / "meta.json").read_text())
    assert meta["cache_format_version"] == 1
    assert meta["has_poi"] is False
    assert Path(meta["graphml_path"]).name == "graph.graphml"

    grid1 = np.load(cache_dir / "grid.npz")
    lat1 = grid1["lat"].astype(float)
    lon1 = grid1["lon"].astype(float)

    # Remove the original source graph and ensure we can still build from the cache.
    graphml.unlink()

    res2 = runner.invoke(get_command(app), ["substrate", "build", "--config", str(cfg_path)])
    assert res2.exit_code == 0, res2.stdout

    grid2 = np.load(cache_dir / "grid.npz")
    np.testing.assert_allclose(lat1, grid2["lat"].astype(float))
    np.testing.assert_allclose(lon1, grid2["lon"].astype(float))
