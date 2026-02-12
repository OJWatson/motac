from __future__ import annotations

import json

import typer

from .._app import data_app


@data_app.command("chicago-load")
def data_chicago_load(
    config: str = typer.Option(
        ..., "--config", help="Path to Chicago raw loader JSON config."
    ),
) -> None:
    """Load Chicago raw contract (v1) and print a small JSON summary."""

    from ...configs import ChicagoRawConfig
    from ...loaders.chicago import load_y_obs_matrix

    cfg = ChicagoRawConfig.from_json(config)
    loaded = load_y_obs_matrix(path=cfg.path, mobility_path=cfg.mobility_path)

    payload = {
        "meta": loaded.meta,
        "y_obs_shape": [int(x) for x in loaded.y_obs.shape],
    }
    typer.echo(json.dumps(payload))
