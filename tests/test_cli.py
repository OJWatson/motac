from __future__ import annotations

import json

import numpy as np
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


def test_sim_forecast_observed_roundtrip_csv(tmp_path) -> None:
    # Tiny end-to-end smoke: write y_obs.csv -> run CLI -> parse JSON -> check shapes.
    y_obs = np.zeros((3, 8), dtype=int)
    y_obs[0, 2] = 1
    y_obs[1, 3] = 2

    path = tmp_path / "y_obs.csv"
    np.savetxt(path, y_obs, fmt="%d", delimiter=",")

    runner = CliRunner()
    res = runner.invoke(
        get_command(app),
        [
            "sim",
            "forecast-observed",
            "--y-obs",
            str(path),
            "--horizon",
            "2",
            "--n-paths",
            "5",
            "--seed",
            "123",
            "--p-detect",
            "0.7",
            "--false-rate",
            "0.2",
            "--n-lags",
            "3",
            "--beta",
            "1.0",
            "--maxiter",
            "10",
        ],
    )

    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert set(payload.keys()) == {"fit", "predict"}

    fit = payload["fit"]
    assert "mu" in fit and "alpha" in fit
    assert len(fit["mu"]) == 3

    pred = payload["predict"]
    q = pred["q"]
    mean = np.asarray(pred["mean"], dtype=float)
    quantiles = np.asarray(pred["quantiles"], dtype=float)

    assert len(q) == quantiles.shape[0]
    assert mean.shape == (3, 2)
    assert quantiles.shape[1:] == (3, 2)


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
