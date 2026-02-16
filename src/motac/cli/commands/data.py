from __future__ import annotations

import json

import typer

from .._app import data_app


@data_app.command("chicago-load")
def data_chicago_load(
    config: str = typer.Option(..., "--config", help="Path to Chicago raw loader JSON config."),
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


@data_app.command("acled-load")
def data_acled_load(
    config: str = typer.Option(
        ..., "--config", help="Path to ACLED events CSV loader JSON config."
    ),
) -> None:
    """Load ACLED events CSV (placeholder schema) and print a small JSON summary."""

    from ...configs import AcledEventsCsvConfig
    from ...loaders.acled import load_acled_events_csv

    cfg = AcledEventsCsvConfig.from_json(config)
    loaded = load_acled_events_csv(
        path=cfg.path,
        mobility_path=cfg.mobility_path,
        value=cfg.value,
    )

    payload = {
        "meta": loaded.meta,
        "y_obs_shape": [int(x) for x in loaded.y_obs.shape],
    }
    typer.echo(json.dumps(payload))


@data_app.command("ingest-events-jsonl")
def data_ingest_events_jsonl(
    input_path: str = typer.Option(
        ..., "--input", help="Path to raw events JSONL (one object per line)."
    ),
    output_path: str = typer.Option(
        ..., "--output", help="Path to write canonical events parquet."
    ),
) -> None:
    """Ingest raw JSONL events into the canonical events table (Parquet)."""

    from ...ingestion import ingest_jsonl_to_canonical_table, write_canonical_events_parquet

    tbl = ingest_jsonl_to_canonical_table(input_path)
    write_canonical_events_parquet(tbl, output_path)

    payload = {
        "n_events": int(tbl.num_rows),
        "schema": str(tbl.schema),
        "output_path": output_path,
    }
    typer.echo(json.dumps(payload))
