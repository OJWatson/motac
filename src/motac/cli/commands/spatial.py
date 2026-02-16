from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from .._app import spatial_app

_GRID_OPT = typer.Option(
    ...,
    "--grid",
    help=("Path to a grid.npz file, or a substrate cache directory containing grid.npz."),
)
_LON_OPT = typer.Option(..., "--lon", help="Longitude (WGS84).")
_LAT_OPT = typer.Option(..., "--lat", help="Latitude (WGS84).")


def _load_grid(path: str | Path):
    from ...substrate.types import Grid

    p = Path(path)
    grid_path = p / "grid.npz" if p.is_dir() else p
    if not grid_path.exists():
        raise typer.BadParameter(f"grid file not found: {grid_path}")

    try:
        npz = np.load(grid_path, allow_pickle=True)
    except Exception as e:  # pragma: no cover
        raise typer.BadParameter(f"failed to load grid npz: {grid_path}") from e

    missing = [k for k in ("lat", "lon", "cell_size_m") if k not in npz]
    if missing:
        raise typer.BadParameter(f"invalid grid.npz (missing keys: {missing}): {grid_path}")

    return Grid(
        lat=np.asarray(npz["lat"], dtype=float),
        lon=np.asarray(npz["lon"], dtype=float),
        cell_size_m=float(np.asarray(npz["cell_size_m"]).ravel()[0]),
    )


@spatial_app.command("cell-id")
def spatial_cell_id(
    grid: str = _GRID_OPT,
    lon: float = _LON_OPT,
    lat: float = _LAT_OPT,
) -> None:
    """Map a lon/lat point to a regular grid cell id.

    Returns -1 if the point is outside the grid.

    This is a thin wrapper around :class:`motac.spatial.lookup.GridCellLookup`.
    """

    from ...spatial.lookup import GridCellLookup

    g = _load_grid(grid)
    lookup = GridCellLookup.from_grid(g)
    cid = lookup.lonlat_to_cell_id(lon=float(lon), lat=float(lat))
    typer.echo(str(int(cid)))
