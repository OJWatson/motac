from __future__ import annotations

import typer

from .._app import substrate_app


@substrate_app.command("build")
def substrate_build(
    config: str = typer.Option(..., "--config", help="Path to substrate JSON config."),
) -> None:
    """Build (and optionally cache) the road-constrained substrate."""

    from ...substrate import SubstrateBuilder, SubstrateConfig

    cfg = SubstrateConfig.from_json(config)
    substrate = SubstrateBuilder(cfg).build()

    typer.echo(f"grid_cells={len(substrate.grid.lat)}")
    nnz = substrate.neighbours.travel_time_s.nnz
    shape = substrate.neighbours.travel_time_s.shape
    typer.echo(f"neighbours_nnz={nnz} shape={shape}")
    if substrate.poi is None:
        typer.echo("poi=disabled")
    else:
        typer.echo(f"poi_features={substrate.poi.x.shape[1]}")
