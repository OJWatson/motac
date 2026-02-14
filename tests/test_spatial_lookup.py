from __future__ import annotations

import numpy as np
import pytest

from motac.spatial.lookup import GridCellLookup
from motac.substrate.builder import build_grid_from_lonlat_bounds


def test_lonlat_to_cell_id_matches_centroid_order():
    grid = build_grid_from_lonlat_bounds(
        lon_min=-0.2,
        lon_max=0.2,
        lat_min=51.4,
        lat_max=51.6,
        cell_size_m=5_000.0,
    )

    lu = GridCellLookup.from_grid(grid)

    # Grid ids are in ravel order already; centroids must map back to their own index.
    cids = lu.lonlat_to_cell_id(grid.lon, grid.lat)
    assert isinstance(cids, np.ndarray)
    assert cids.shape == (grid.lon.shape[0],)
    assert np.array_equal(cids, np.arange(grid.lon.shape[0], dtype=int))


def test_lonlat_to_cell_id_outside_returns_minus_one():
    grid = build_grid_from_lonlat_bounds(
        lon_min=-0.1,
        lon_max=0.1,
        lat_min=51.4,
        lat_max=51.6,
        cell_size_m=10_000.0,
    )
    lu = GridCellLookup.from_grid(grid)

    # Far outside bounds.
    assert lu.lonlat_to_cell_id(lon=10.0, lat=0.0) == -1

    # Array input: mixture of inside/outside.
    lon = np.asarray([grid.lon[0], 10.0])
    lat = np.asarray([grid.lat[0], 0.0])
    out = lu.lonlat_to_cell_id(lon=lon, lat=lat)
    assert np.array_equal(out, np.asarray([0, -1], dtype=int))


def test_from_grid_rejects_non_rectangular_grid():
    # Construct an invalid grid by dropping an interior point (creating a hole).
    grid = build_grid_from_lonlat_bounds(
        lon_min=-0.3,
        lon_max=0.3,
        lat_min=51.2,
        lat_max=51.8,
        cell_size_m=5_000.0,
    )

    lu_full = GridCellLookup.from_grid(grid)
    assert lu_full.nx >= 3 and lu_full.ny >= 3

    # Drop a definitely-interior index (ix=1, iy=1) so (nx, ny) recovered from
    # remaining points stays the same, but n_cells no longer matches nx*ny.
    interior = 1 * lu_full.nx + 1
    keep = np.ones(grid.lon.shape[0], dtype=bool)
    keep[int(interior)] = False
    bad = type(grid)(lat=grid.lat[keep], lon=grid.lon[keep], cell_size_m=grid.cell_size_m)

    with pytest.raises(ValueError, match="full regular rectangle"):
        GridCellLookup.from_grid(bad)
