from __future__ import annotations

import numpy as np
from click.testing import CliRunner
from typer.main import get_command

from motac.cli import app
from motac.spatial.grid_builder import LonLatBounds, build_regular_grid
from motac.spatial.lookup import GridCellLookup


def test_cli_spatial_cell_id_smoke(tmp_path) -> None:
    # Build a tiny regular grid, persist it in the same on-disk format as the
    # substrate cache bundle uses (grid.npz), and check the CLI maps a centroid
    # to the expected cell id.
    bounds = LonLatBounds(
        lon_min=-0.100,
        lon_max=-0.098,
        lat_min=51.500,
        lat_max=51.502,
    )
    grid = build_regular_grid(bounds, cell_size_m=100.0)

    grid_path = tmp_path / "grid.npz"
    np.savez(
        grid_path,
        lat=np.asarray(grid.lat, dtype=float),
        lon=np.asarray(grid.lon, dtype=float),
        cell_size_m=np.asarray([float(grid.cell_size_m)], dtype=float),
    )

    lookup = GridCellLookup.from_grid(grid)
    lon0 = float(grid.lon[0])
    lat0 = float(grid.lat[0])
    expected = int(lookup.lonlat_to_cell_id(lon=lon0, lat=lat0))

    runner = CliRunner()
    res = runner.invoke(
        get_command(app),
        ["spatial", "cell-id", "--grid", str(grid_path), "--lon", str(lon0), "--lat", str(lat0)],
    )

    assert res.exit_code == 0, res.stdout
    assert int(res.stdout.strip()) == expected
