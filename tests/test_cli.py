from __future__ import annotations

from click.testing import CliRunner
from typer.main import get_command

from motac.cli import app


def test_version_command() -> None:
    runner = CliRunner()
    res = runner.invoke(get_command(app), ["version"])
    assert res.exit_code == 0
    assert res.stdout.strip() != ""


def test_sim_fit_observed_help() -> None:
    runner = CliRunner()
    res = runner.invoke(get_command(app), ["sim", "fit-observed", "--help"])
    assert res.exit_code == 0
    assert "Fit (mu, alpha)" in res.stdout


def test_sim_forecast_observed_help() -> None:
    runner = CliRunner()
    res = runner.invoke(get_command(app), ["sim", "forecast-observed", "--help"])
    assert res.exit_code == 0
    assert "Observed-only forecast" in res.stdout


def test_substrate_build_command(tmp_path) -> None:
    import json

    import networkx as nx
    import osmnx as ox

    graphml = tmp_path / "tiny.graphml"
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    G.add_node(0, x=-0.1, y=51.5)
    G.add_node(1, x=-0.099, y=51.5)
    G.add_edge(0, 1, key=0, travel_time=60.0, length=100.0)
    G.add_edge(1, 0, key=0, travel_time=60.0, length=100.0)
    ox.save_graphml(G, filepath=graphml)

    poi_path = tmp_path / "pois.geojson"
    poi_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {"type": "Point", "coordinates": [-0.1, 51.5]},
                    }
                ],
            }
        )
    )

    cache_dir = tmp_path / "cache"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "graphml_path": str(graphml),
                "cell_size_m": 100.0,
                "max_travel_time_s": 120.0,
                "poi_geojson_path": str(poi_path),
                "cache_dir": str(cache_dir),
            }
        )
    )

    runner = CliRunner()
    res = runner.invoke(get_command(app), ["substrate", "build", "--config", str(cfg_path)])
    assert res.exit_code == 0, res.stdout
    assert "grid_cells=" in res.stdout
    assert (cache_dir / "grid.npz").exists()
