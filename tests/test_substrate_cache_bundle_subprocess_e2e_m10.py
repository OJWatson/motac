from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import networkx as nx
import osmnx as ox


def _run_motac_cli(*args: str) -> subprocess.CompletedProcess[str]:
    # Execute the CLI in a separate Python process for a true end-to-end smoke.
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "from motac.cli import main; "
            "sys.argv=['motac', *sys.argv[1:]]; "
            "main()"
        ),
        *args,
    ]
    return subprocess.run(cmd, check=False, text=True, capture_output=True)


def test_substrate_cache_bundle_v1_subprocess_roundtrip(tmp_path: Path) -> None:
    """End-to-end smoke: subprocess CLI build writes a bundle and reloads from it."""

    graphml = tmp_path / "tiny.graphml"
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    G.add_node(0, x=-0.1, y=51.5)
    G.add_node(1, x=-0.099, y=51.5)
    G.add_edge(0, 1, key=0, travel_time=60.0, length=100.0)
    G.add_edge(1, 0, key=0, travel_time=60.0, length=100.0)
    ox.save_graphml(G, filepath=graphml)

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

    res1 = _run_motac_cli("substrate", "build", "--config", str(cfg_path))
    assert res1.returncode == 0, res1.stdout + res1.stderr
    assert "poi=disabled" in res1.stdout

    # The cache bundle should be self-contained.
    assert (cache_dir / "graph.graphml").exists()
    assert (cache_dir / "grid.npz").exists()
    assert (cache_dir / "neighbours.npz").exists()
    assert (cache_dir / "meta.json").exists()

    # Remove the original source graph and ensure we can still build from the cache.
    graphml.unlink()

    res2 = _run_motac_cli("substrate", "build", "--config", str(cfg_path))
    assert res2.returncode == 0, res2.stdout + res2.stderr
